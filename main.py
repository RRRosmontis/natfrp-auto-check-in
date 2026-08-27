import logging
import os
import sys
import time
import random
import shutil

import cv2
import numpy as np

from playwright.sync_api import sync_playwright, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import Error as PlaywrightError

# 将 SakuraFrpQiandao 加入搜索路径，复用其模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SakuraFrpQiandao"))

from config import Config
from captcha_handler import CaptchaHandler, ModelApiError
from human_simulator import HumanSimulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('checkin.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)


def get_gap_offset(bg_path: str, fullbg_path: str) -> int:
    """OpenCV 滑块缺口检测"""
    bg = cv2.imread(bg_path, 0)
    fullbg = cv2.imread(fullbg_path, 0)

    diff = cv2.absdiff(fullbg, bg)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("轮廓数量不足，滑块识别失败")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    rects = [cv2.boundingRect(c) for c in sorted_contours]
    target = max(rects, key=lambda r: r[0])
    x, y, w, h = target
    logger.info(f"滑块缺口坐标: x={x}, y={y}, w={w}, h={h}")
    return x


def drag_slider(page: Page, slider, canvas_bg, gap_x: int):
    """模拟人类拖动滑块，根据 canvas 缩放比例校正距离"""
    # 计算 canvas 显示宽度与实际截图宽度的比例
    canvas_box = canvas_bg.bounding_box()
    bg_img = cv2.imread("bg.png", 0)
    img_w = bg_img.shape[1]  # 截图宽度（像素）
    display_w = canvas_box['width']  # CSS 显示宽度
    scale = display_w / img_w if img_w > 0 else 1.0
    distance = int(gap_x * scale)
    logger.info(f"Canvas 缩放比例: {scale:.3f} (img={img_w}px, display={display_w:.1f}px), 校正距离: {distance}px")

    slider_box = slider.bounding_box()
    start_x = slider_box['x'] + slider_box['width'] / 2
    start_y = slider_box['y'] + slider_box['height'] / 2

    logger.info(f"滑块起点: ({start_x:.0f}, {start_y:.0f})")

    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.wait_for_timeout(random.randint(200, 400))

    # 分段拖动到目标位置过一点，再回退微调（模拟人类）
    page.mouse.move(start_x + distance + random.randint(2, 8), start_y + random.uniform(-1, 1))
    page.wait_for_timeout(random.randint(300, 600))
    page.mouse.move(start_x + distance - random.randint(2, 5), start_y)
    page.wait_for_timeout(random.randint(200, 400))
    page.mouse.move(start_x + distance - random.randint(5, 8), start_y)
    page.mouse.up()

    logger.info("滑块拖动完成")


def solve_slider_captcha(page: Page) -> bool:
    """处理滑块验证码（OpenCV 方案）"""
    logger.info("检测到滑块验证码，使用 OpenCV 方案处理...")
    try:
        # 截图含缺口的背景
        bg = page.query_selector('.geetest_canvas_bg')
        if not bg:
            logger.error("未找到滑块背景 canvas")
            return False
        bg.screenshot(path="bg.png")

        # 显示完整背景并截图
        page.evaluate(
            "document.querySelector('.geetest_canvas_fullbg').style.display = 'block';"
        )
        fullbg = page.query_selector('.geetest_canvas_fullbg')
        fullbg.screenshot(path="fullbg.png")
        page.evaluate(
            "document.querySelector('.geetest_canvas_fullbg').style.display = 'none';"
        )

        gap_x = get_gap_offset("bg.png", "fullbg.png")

        slider = page.query_selector('.geetest_slider_button')
        if not slider:
            logger.error("未找到滑块按钮")
            return False

        drag_slider(page, slider, bg, gap_x)
        time.sleep(2)

        # 检查成功文案或 slider 消失
        try:
            page.wait_for_selector('.geetest_slider_button', state='detached', timeout=3000)
            logger.info("滑块已消失，验证通过")
            return True
        except PlaywrightTimeoutError:
            logger.warning("滑块未消失，可能未通过")
            return False

    except Exception as e:
        logger.error(f"滑块验证处理出错: {e}", exc_info=True)
        return False


def detect_captcha_type(page: Page) -> str:
    """检测验证码类型: 'slider' | 'grid' | 'none'"""
    try:
        slider = page.query_selector('.geetest_slider_button')
        if slider and slider.is_visible():
            return 'slider'
    except Exception:
        pass

    try:
        grid = page.query_selector('.geetest_tip_img')
        if grid and grid.is_visible():
            return 'grid'
    except Exception:
        pass

    return 'none'


class CheckInBot:
    def __init__(self, config: Config):
        self.config = config
        self.captcha_handler = CaptchaHandler(config)
        self.simulator = HumanSimulator()
        self.max_retries = config.max_retries

    def run(self):
        headless = os.getenv('CI') == 'true' or os.getenv('HEADLESS', 'false').lower() == 'true'

        with sync_playwright() as playwright:
            browser = None
            try:
                launch_opts = {
                    "headless": headless,
                    "args": [
                        "--window-size=1280,800",
                        "--disable-blink-features=AutomationControlled",
                        "--no-proxy-server",
                        "--lang=zh-CN",
                        "--disable-gpu",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                }
                chrome_path = self._resolve_chrome()
                if chrome_path:
                    launch_opts["executable_path"] = chrome_path

                browser = playwright.chromium.launch(**launch_opts)
                context = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    locale="zh-CN",
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
                )
                page = context.new_page()
                page.set_default_timeout(20000)

                self._login(page)
                self._navigate_dashboard(page)
                self._do_checkin(page)
                logger.info("签到流程完成")

            except ModelApiError:
                logger.error("模型 API 调用失败，终止")
            except Exception as e:
                logger.error(f"执行出错: {e}", exc_info=True)
            finally:
                if browser and headless:
                    browser.close()

    def _resolve_chrome(self):
        if self.config.chrome_binary_path and os.path.exists(self.config.chrome_binary_path):
            return self.config.chrome_binary_path
        for exe in ("google-chrome", "google-chrome-stable", "chrome", "chromium"):
            found = shutil.which(exe)
            if found:
                return found
        return None

    def _login(self, page: Page):
        url = "https://www.natfrp.com/user/"
        logger.info(f"导航到登录页: {url}")
        page.goto(url, wait_until="domcontentloaded")

        user_input = page.locator("#username")
        pass_input = page.locator("#password")
        user_input.wait_for(state="visible")
        pass_input.wait_for(state="visible")

        logger.info("输入登录凭据...")
        user_input.fill("")
        self.simulator.type_text(user_input, self.config.sakurafrp_user)
        pass_input.fill("")
        self.simulator.type_text(pass_input, self.config.sakurafrp_pass)

        login_btn = page.locator("#login")
        login_btn.wait_for(state="visible")
        login_btn.click()
        logger.info("登录成功")

    def _navigate_dashboard(self, page: Page):
        self.simulator.random_sleep(3, 5)
        self._dismiss_age_check(page)

    def _dismiss_age_check(self, page: Page) -> bool:
        """尝试多种方式解除年龄确认弹窗"""
        # 方式1: 检测 adult-check 弹窗，点击"是"按钮
        try:
            adult_check = page.locator(".adult-check")
            if adult_check.count() > 0 and adult_check.first.is_visible():
                logger.info("检测到年龄确认弹窗...")
                yes_btn = page.locator(".adult-check button").first
                yes_btn.wait_for(state="visible", timeout=3000)
                yes_btn.click(force=True)
                logger.info("已点击年龄确认按钮（force click）")
                time.sleep(1)
                return True
        except Exception:
            pass

        # 方式2: 匹配文本"是，我已满18岁"
        try:
            age = page.locator("div.yes a", has_text="是，我已满18岁")
            age.wait_for(state="visible", timeout=3000)
            age.click()
            logger.info("已处理年龄确认（文本匹配）")
            time.sleep(1)
            return True
        except PlaywrightTimeoutError:
            pass

        logger.info("无需年龄确认")
        return False

    def _already_signed(self, page: Page) -> bool:
        try:
            page.wait_for_selector("text=今天已经签到过啦", timeout=3000)
            return True
        except PlaywrightTimeoutError:
            return False

    def _do_checkin(self, page: Page):
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"签到尝试 {attempt}/{self.max_retries}")

            # 先检查是否已签到
            if self._already_signed(page):
                logger.info("今日已签到")
                return

            btn = None
            try:
                btn = page.locator("button", has_text="点击这里签到")
                btn.wait_for(state="visible", timeout=5000)
            except PlaywrightTimeoutError:
                # 等久一点再检查一次
                time.sleep(2)
                if self._already_signed(page):
                    logger.info("今日已签到（延迟检测）")
                    return
                logger.error("未找到签到按钮，也未检测到已签到标识")
                return

            logger.info("点击签到按钮...")
            self._dismiss_age_check(page)  # 二次确认，防止弹窗阻挡点击
            btn.click(force=True)
            self.simulator.random_sleep(2, 4)

            captcha_type = detect_captcha_type(page)
            logger.info(f"验证码类型: {captcha_type}")

            if captcha_type == 'slider':
                if solve_slider_captcha(page):
                    page.reload(wait_until="domcontentloaded")
                    time.sleep(3)
                    if self._already_signed(page):
                        logger.info("滑块验证 + 签到成功")
                        return
                else:
                    page.reload(wait_until="domcontentloaded")
                    time.sleep(3)
                    continue

            elif captcha_type == 'grid':
                try:
                    self.captcha_handler.handle_geetest_captcha(page)
                    page.reload(wait_until="domcontentloaded")
                    time.sleep(3)
                    if self._already_signed(page):
                        logger.info("九宫格验证 + 签到成功")
                        return
                except ModelApiError:
                    raise
                except Exception as e:
                    logger.error(f"九宫格处理出错: {e}")

            elif captcha_type == 'none':
                time.sleep(3)
                if self._already_signed(page):
                    logger.info("签到成功（无验证码）")
                    return

            page.reload(wait_until="domcontentloaded")
            time.sleep(3)

        logger.warning("已达最大重试次数")


def main():
    try:
        config = Config.from_env()
        logger.info(f"使用账户: {config.sakurafrp_user}")
        bot = CheckInBot(config)
        bot.run()
    except ValueError as e:
        logger.error(f"配置错误: {e}")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()
