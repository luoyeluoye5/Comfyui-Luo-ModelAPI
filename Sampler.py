import requests
from comfy.comfy_types import IO
from comfy_api.input_impl import VideoFromFile
from io import BytesIO
from volcenginesdkarkruntime import Ark
import time
import base64
from PIL import Image
import io
import torch
import numpy as np
from volcenginesdkarkruntime import Ark

class LuoT2V:
    """
    文生视频
    """
    modellist = ["doubao-seedance-1-0-pro-250528"]
    resolutionlist = ["480p", "720p", "1080p"]
    ratiolist = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]
    durationlist = [5, 10]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "请输入详细的画面描述，例如：戴着帽子的老爷爷面带微笑往前走。",
                    },
                ),
                "model": ((s.modellist,)), 
                # "model": ("STRING", {"default": "请在此处粘贴您的火山引擎模型名称"}),
                "resolution": ((s.resolutionlist,)),  # 视频分辨率
                "ratio": ((s.ratiolist,)),  # 生成视频的宽高比例
                "duration": ((s.durationlist,)),  # 生成视频时长
                "framepersecond": (
                    "INT",
                    {"default": 16, "min": 16, "max": 24},
                ),  # 帧率，即一秒时间内视频画面数量
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF},
                ),  # 种子
                "APIKey": ("STRING", {"default": "请在此处粘贴您的火山引擎API Key"}),
            },
        }

    # 返回我们目标节点所需要的 IO.VIDEO 类型
    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "Tokens使用量")

    FUNCTION = "load_and_create_video_object"
    CATEGORY = "API Video"

    def load_and_create_video_object(
        self,
        prompt,
        model,
        resolution,
        ratio,
        duration,
        framepersecond,
        seed,
        APIKey,
    ):

        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=APIKey,
        )

        create_result = client.content_generation.tasks.create(
            model=model,
            content=[
                {
                    "type": "text",
                    "text": f"{prompt} --resolution {resolution} --ratio {ratio} --duration {duration} --framepersecond {framepersecond} --watermark  false --seed  {seed}",
                },
            ],
        )
        print(create_result)
        video_url = None
        print("----- polling task status -----")
        task_id = create_result.id
        tokens_from_api = None
        while True:
            get_result = client.content_generation.tasks.get(task_id=task_id)
            status = get_result.status
            if status == "succeeded":
                print("----- task succeeded -----")
                print(get_result)
                video_url = get_result.content.video_url
                tokens_from_api = get_result.usage.completion_tokens
                break
            elif status == "failed":
                print("----- task failed -----")
                print(f"Error: {get_result.error}")
                break
            else:
                print(f"Current status: {status}, Retrying after 10 seconds...")
                time.sleep(10)

        if not video_url:
            raise ValueError("视频 URL 不能为空")

        vid_response = requests.get(video_url)

        return {
            "result": (
                VideoFromFile(BytesIO(vid_response.content)),
                [f"Tokens: {tokens_from_api}"],
            )
        }

class LuoT2I:
    """
    文生图
    """
    def __init__(self):
        pass
    RESOLUTIONS = [
        "1024x1024",
        "864x1152",
        "1152x864",
        "1280x720",
        "720x1280",
        "832x1248",
        "1248x832",
        "1512x648",
    ]
    modellist =[
        "doubao-seedream-3-0-t2i-250415"
    ]
    @classmethod
    def INPUT_TYPES(s):
        """
        返回一个包含所有输入字段配置的字典。
        一些类型（字符串）："MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT"。
        输入类型 "INT", "STRING" 或 "FLOAT" 是节点上字段的特殊值。
        类型可以是一个列表，用于提供下拉选择。

        返回: `dict`:
            - 键 input_fields_group (`string`): 可以是 "required"（必需）、"hidden"（隐藏）或 "optional"（可选）。一个节点类必须有 "required" 属性。
            - 值 input_fields (`dict`): 包含输入字段的配置：
                * 键 field_name (`string`): 入口方法参数的名称。
                * 值 field_config (`tuple`):
                    + 第一个值是表示字段类型的字符串，或者是一个用于下拉选择的列表。
                    + 第二个值是为 "INT", "STRING" 或 "FLOAT" 类型准备的配置字典。
        """
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "请输入详细的画面描述，例如：一只可爱的猫咪，背景是星空。"}),
                # "model": ("STRING", {"default": "请在此处粘贴您的火山引擎模型名称"}),
                "model": ((s.modellist,)), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffff}),
                # "mode": (["固定", "递增", "递减", "随机"],),
                "APIKey": ("STRING", {"default": "请在此处粘贴您的火山引擎API Key"}),#ae01b368-48a0-48df-9e49-5173741a1ef9
                # "resolution" 是输入参数的内部名称
                # (s.RESOLUTIONS,) 是一个元组，第一个元素是包含所有选项字符串的列表
                # 这会告诉 ComfyUI 创建一个下拉框
                "size": (s.RESOLUTIONS,),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "test"


    CATEGORY = "LuoImage"

    def test(self, prompt,model,seed, APIKey,size):
            
        client = Ark(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key=APIKey,
        )

        imagesResponse = client.images.generate(
            model=model,
            prompt=prompt,
            response_format="b64_json",
            size=size,
            seed =seed,
        )

        b64_string = imagesResponse.data[0].b64_json
        # 解码Base64数据
        # Base64字符串解码后得到的是二进制数据（bytes）
        image_bytes = base64.b64decode(b64_string)

        # 将解码后的二进制数据读入Pillow
        # 使用io.BytesIO将内存中的二进制数据模拟成一个文件对象
        image_data = io.BytesIO(image_bytes)

        # 使用Pillow的Image模块打开这个“文件”
        image = Image.open(image_data)
        
         # --- 新增的转换代码 ---
        # 1. 将 Pillow Image 转换为 NumPy 数组
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # 2. 将 NumPy 数组转换为 PyTorch Tensor
        image_tensor = torch.from_numpy(image_np)
        
        # 3. 增加一个批次维度 (batch dimension)，ComfyUI 需要这个维度
        #    转换后的形状为 [1, 高度, 宽度, 通道数]
        image_tensor = image_tensor.unsqueeze(0)
        # --- 转换结束 ---

        # 返回符合 ComfyUI 格式的 Tensor
        return (image_tensor,)

    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# 注册节点
NODE_CLASS_MAPPINGS = {"文生视频模型API_火山方舟": LuoT2V,"文生图片模型API_火山方舟":LuoT2I}
