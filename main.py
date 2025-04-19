import os
import json
import tomllib
import traceback
from typing import Dict, List

import aiohttp
from loguru import logger

from WechatAPI import WechatAPIClient
from utils.decorators import *
from utils.plugin_base import PluginBase


class SiliconFlow(PluginBase):
    description = "硅基流动API插件"
    author = "老夏的金库"
    version = "1.0.0"
    is_ai_platform = True  # 标记为 AI 平台插件

    def __init__(self):
        super().__init__()

        try:
            # 读取插件配置
            config_path = os.path.join(os.path.dirname(__file__), "config.toml")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            # 获取SiliconFlow配置
            plugin_config = config.get("SiliconFlow", {})
            self.enable = plugin_config.get("enable", False)
            self.handle_all_messages = plugin_config.get("handle_all_messages", False)
            self.api_key = plugin_config.get("api-key", "")
            self.base_url = plugin_config.get("base-url", "https://api.siliconflow.cn/v1")

            # 获取模型配置
            self.default_model = plugin_config.get("default-model", "Qwen/QwQ-32B")
            self.available_models = plugin_config.get("available-models", ["Qwen/QwQ-32B"])

            # 获取命令配置
            self.commands = plugin_config.get("commands", ["硅基", "sf", "SiliconFlow"])

            # 获取功能配置
            self.http_proxy = plugin_config.get("http-proxy", "")

            # 获取高级设置
            self.max_tokens = plugin_config.get("max_tokens", 4096)
            self.temperature = plugin_config.get("temperature", 0.7)
            self.top_p = plugin_config.get("top_p", 0.7)
            self.top_k = plugin_config.get("top_k", 50)
            self.frequency_penalty = plugin_config.get("frequency_penalty", 0.5)

            logger.success("SiliconFlow插件初始化成功")

        except Exception as e:
            logger.error(f"SiliconFlow插件初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.enable = False

    async def call_silicon_flow_api(self, messages: List[Dict[str, str]], model: str = None) -> str:
        """调用硅基流动API

        Args:
            messages: 消息列表
            model: 模型名称，如果为None则使用默认模型

        Returns:
            str: API返回的文本内容
        """
        try:
            # 使用默认模型（如果未指定）
            if model is None:
                model = self.default_model

            # 构建请求数据
            data = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.frequency_penalty,
                "n": 1,
                "response_format": {
                    "type": "text"
                }
            }

            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # 设置代理
            proxy = self.http_proxy if self.http_proxy else None

            # 发送请求
            url = f"{self.base_url}/chat/completions"
            logger.info(f"调用硅基流动API: {url}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=data,
                    proxy=proxy
                ) as response:
                    # 获取响应
                    response_text = await response.text()

                    # 检查响应状态
                    if response.status != 200:
                        logger.error(f"硅基流动API返回错误: {response.status} {response_text}")
                        return f"API调用失败: {response.status} {response_text}"

                    # 解析响应
                    try:
                        response_json = json.loads(response_text)

                        # 提取回复内容
                        if "choices" in response_json and len(response_json["choices"]) > 0:
                            message = response_json["choices"][0].get("message", {})
                            content = message.get("content", "")
                            return content
                        else:
                            logger.error(f"硅基流动API返回格式错误: {response_json}")
                            return "API返回格式错误"
                    except json.JSONDecodeError:
                        logger.error(f"硅基流动API返回非JSON格式: {response_text}")
                        return f"API返回非JSON格式: {response_text}"

        except Exception as e:
            logger.error(f"调用硅基流动API失败: {str(e)}")
            logger.error(traceback.format_exc())
            return f"调用API失败: {str(e)}"

    @on_text_message(priority=20)
    async def handle_text_message(self, bot: WechatAPIClient, message: dict):
        """处理文本消息"""
        if not self.enable:
            return True  # 插件未启用，允许后续插件处理

        content = message["Content"]
        from_wxid = message["FromWxid"]

        # 检查API密钥是否已设置
        if not self.api_key:
            logger.warning("硅基流动API密钥未设置，请在配置文件中设置api-key")
            return True  # 允许后续插件处理

        # 检查是否以命令开头
        for command in self.commands:
            if content.startswith(command):
                # 提取参数
                args = content[len(command):].strip()

                # 检查是否有消息内容
                if not args:
                    await bot.send_text_message(from_wxid, "请输入要发送给硅基流动的消息")
                    return False

                # 构建消息
                messages = [
                    {
                        "role": "user",
                        "content": args
                    }
                ]

                # 发送等待消息
                await bot.send_text_message(from_wxid, "正在请求硅基流动API，请稍候...")

                # 调用API
                response = await self.call_silicon_flow_api(messages)

                # 发送回复
                await bot.send_text_message(from_wxid, response)

                return False  # 阻止后续插件处理

        # 检查是否是私聊消息
        is_private_chat = True

        # 检查多种可能的群聊标识
        if "RoomWxid" in message and message["RoomWxid"]:
            is_private_chat = False
        elif "FromWxid" in message and "@chatroom" in message["FromWxid"]:
            is_private_chat = False
        elif "chat_id" in message and "@chatroom" in message.get("chat_id", ""):
            is_private_chat = False

        logger.debug(f"消息类型检测: is_private_chat={is_private_chat}, message={message}")

        # 如果是私聊消息，则直接处理
        if is_private_chat and content.strip():
            logger.info(f"私聊模式，处理消息: {content}")

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # 调用API
            response = await self.call_silicon_flow_api(messages)

            # 发送回复
            await bot.send_text_message(from_wxid, response)

            return False  # 阻止后续插件处理
        # 如果是群聊消息且设置了处理所有消息，则直接处理
        elif not is_private_chat and self.handle_all_messages and content.strip():
            logger.info(f"群聊处理所有消息模式，收到消息: {content}")

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # 调用API
            response = await self.call_silicon_flow_api(messages)

            # 发送回复
            await bot.send_text_message(from_wxid, response)

            return False  # 阻止后续插件处理

        return True  # 不符合处理条件，允许后续插件处理

    @on_at_message(priority=20)
    async def handle_at_message(self, bot: WechatAPIClient, message: dict):
        """处理@消息"""
        if not self.enable:
            return True  # 插件未启用，允许后续插件处理

        # 检查API密钥是否已设置
        if not self.api_key:
            logger.warning("硅基流动API密钥未设置，请在配置文件中设置api-key")
            return True  # 允许后续插件处理

        content = message["Content"]
        from_wxid = message["FromWxid"]

        # 提取实际内容（去除@部分）
        # 注意：这里假设Content中包含了@信息，实际情况可能需要更复杂的处理
        # 简单处理：查找第一个空格后的内容
        if " " in content:
            content = content.split(" ", 1)[1].strip()
        else:
            content = content.strip()

        # 检查是否以命令开头
        for command in self.commands:
            if content.startswith(command):
                # 提取参数
                args = content[len(command):].strip()

                # 检查是否有消息内容
                if not args:
                    await bot.send_text_message(from_wxid, "请输入要发送给硅基流动的消息")
                    return False

                # 构建消息
                messages = [
                    {
                        "role": "user",
                        "content": args
                    }
                ]

                # 发送等待消息
                await bot.send_text_message(from_wxid, "正在请求硅基流动API，请稍候...")

                # 调用API
                response = await self.call_silicon_flow_api(messages)

                # 发送回复
                await bot.send_text_message(from_wxid, response)

                return False  # 阻止后续插件处理

        # 如果没有命令前缀，直接处理@消息
        if content.strip():
            logger.info(f"@消息模式，处理消息: {content}")

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            # 调用API
            response = await self.call_silicon_flow_api(messages)

            # 发送回复
            await bot.send_text_message(from_wxid, response)

            return False  # 阻止后续插件处理

        return True  # 消息为空，允许后续插件处理
