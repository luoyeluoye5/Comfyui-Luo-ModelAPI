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
import json
import numpy as np
from volcenginesdkarkruntime import Ark
from volcengine.visual.VisualService import VisualService

class LuoT2V:
    """
    文生视频
    """
    def __init__(self):
        print("LuoT2V 节点已初始化！") # 添加这一行
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
         print("LuoT2I 节点已初始化！") # 添加这一行

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
    modellist = ["doubao-seedream-3-0-t2i-250415", "jimeng_high_aes_general_v21_L"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "请输入详细的画面描述，例如：一只可爱的猫咪，背景是星空。",
                    },
                ),
                "model": ((s.modellist,)),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "APIKey": (
                    "STRING",
                    {"default": "请在此处粘贴您的火山引擎AccessKeyID"},
                ),
                "width":  ("INT", {"default": 512, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "height": ("INT", {"default": 512, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "SecretAccessKey": (
                    "STRING",
                    {"default": "请在此处粘贴您的SecretAccessKey(调用即梦模型才需要)"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "test"

    CATEGORY = "LuoImage"

    def FangZhouAPI(self,prompt, model, seed, APIKey, width,height, SecretAccessKey):
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
            size= f"{width}x{height}",
            seed=seed,
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
    
    def JiMengAPI(self,prompt, model, seed, APIKey, width,height, SecretAccessKey):
        visual_service = VisualService()
        visual_service.set_ak(APIKey)
        visual_service.set_sk(SecretAccessKey)

        form = {
        "req_key": model,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "use_pre_llm": True,
        "use_sr": True,
        "return_url": True,
         }

        reply = visual_service.cv_process(form)
    
        if reply and reply.get("code") == 10000:
          print("\nAPI 调用成功，返回结果：")
          print(json.dumps(reply, indent=4, ensure_ascii=False))
          data = reply.get("data", {})
          image_urls = data.get("image_urls")
          
          response = requests.get(image_urls[0])
          response.raise_for_status()  # 检查请求是否成功

          image_data = BytesIO(response.content)
          pil_image = Image.open(image_data).convert("RGB")  # 确保是RGB格式

          image_np = np.array(pil_image).astype(np.float32) / 255.0
          image_tensor = torch.from_numpy(image_np)[None,]  # 添加 batch 维度
          print(image_tensor)
          return (image_tensor,)
        elif response:
          print(f"\nAPI 调用失败，错误码: {response.get('code')}, 错误信息: {response.get('message')}")
          print(f"原始请求ID: {response.get('request_id')}")
          return (None,)
        else:
          print("\nAPI 调用失败或返回空响应。")
          return (None,)
      
    def test(self, prompt, model, seed, APIKey, width,height, SecretAccessKey):
      if "doubao" in model:
        return self.FangZhouAPI(prompt, model, seed, APIKey, width,height, SecretAccessKey)
      else:
        return self.JiMengAPI(prompt, model, seed, APIKey, width,height, SecretAccessKey)

    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# 注册节点
NODE_CLASS_MAPPINGS = {
    "文生视频模型API_火山引擎": LuoT2V,
    "文生图片模型API_火山引擎": LuoT2I,
}
