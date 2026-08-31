"""Optional WeCom webhook and SMTP notifications (best-effort, non-blocking)."""

from __future__ import annotations

import logging
import smtplib
import threading
from email.message import EmailMessage

import requests

from .config import Settings
from .presentation import label_value


LOG = logging.getLogger(__name__)


class NotificationHub:
    """通知中心：企业微信机器人 + SMTP 邮件两个可选通道，尽力投递、失败不阻断交易。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, title: str, message: str, level: str = "INFO", html: str | None = None) -> None:
        """Best-effort, non-blocking delivery: failures never stop trading safeguards.

        投递在后台守护线程中执行，避免 SMTP / 企业微信的网络延迟阻塞信号生成或成交主流程。
        """
        if not (self.settings.wecom_webhook or (self.settings.smtp_host and self.settings.email_to)):
            return
        threading.Thread(
            target=self._deliver, args=(title, message, level, html),
            daemon=True, name="notification",
        ).start()

    def _deliver(self, title: str, message: str, level: str, html: str | None) -> None:
        """实际投递：先发企业微信（纯文本），再发 SMTP 邮件（可选 HTML 富文本）。"""
        text = f"[{label_value(level)}] {title}\n{message}"
        if self.settings.wecom_webhook:
            try:
                response = requests.post(
                    self.settings.wecom_webhook,
                    json={"msgtype": "text", "text": {"content": text}}, timeout=8,
                )
                response.raise_for_status()
            except Exception as error:
                LOG.warning("企业微信通知发送失败：%s", error)
        if self.settings.smtp_host and self.settings.email_to:
            try:
                email = EmailMessage()
                email["Subject"] = f"A 股量化交易 | {title}"
                email["From"] = self.settings.smtp_username
                email["To"] = self.settings.email_to
                email.set_content(text)
                if html:
                    email.add_alternative(html, subtype="html")
                with smtplib.SMTP_SSL(self.settings.smtp_host, self.settings.smtp_port, timeout=10) as smtp:
                    if self.settings.smtp_username:
                        smtp.login(self.settings.smtp_username, self.settings.smtp_password)
                    smtp.send_message(email)
            except Exception as error:
                LOG.warning("邮件通知发送失败：%s", error)
