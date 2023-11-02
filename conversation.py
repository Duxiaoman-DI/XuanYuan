"""
refer: https://github.com/lm-sys/FastChat/tree/main/fastchat
"""

import dataclasses
from enum import auto, IntEnum
from typing import List, Dict


class SeparatorStyle(IntEnum):
    """Separator styles."""
    ADD_COLON_TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: List[str] = (("USER", "ASSISTANT"),)
    messages: List[List[str]] = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_TWO
    sep: str = "\n"
    sep2: str = None
    stop_str: str = None
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        self.messages[-1][1] = message

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="XuanYuan-Chat",
        system_message="以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)


if __name__ == "__main__":

    conv = get_conv_template("XuanYuan-Chat")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "介绍下你自己")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
