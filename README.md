# Rust Vulkan Video Player
## Introduction
A simple video player based on Vulkan implemented by Rust

## Abilities
- Using `ffmpeg` for video decode
- Using `Vulkano` for video playback
- Using `rodio` for music playback
- Achieving synchronization between video and audio streams

## Usage
```cmd
executable [video_path]
```
If you don't provide the path, the default value is "assets/videos/video.mp4"

## Regrets
- Vulkano doesn't support the `Vulkan Video` extensions, so I have to use ffmpeg to decode the video stream...

## Probabilities
- Maybe I can refactor this program with `Ash`...

## Bug reports
### 1.
If you enable the validation layer, you will see validation message three times like this:
```
Validation Layer[ERROR][VALIDATION]: vkQueueSubmit(): pSubmits[0] command buffer VkCommandBuffer 0x1fc6fe82810 expects VkImage 0x460000000046 (subresource: aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, mipLevel = 0, arrayLayer = 0) to be in layout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR--instead, current layout is VK_IMAGE_LAYOUT_UNDEFINED.
The Vulkan spec states: If a descriptor with type equal to any of VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM, VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, or VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT is accessed as a result of this command, all image subresources identified by that descriptor must be in the image layout identified when the descriptor was written (https://vulkan.lunarg.com/doc/view/1.4.321.1/windows/antora/spec/latest/chapters/drawing.html#VUID-vkCmdDraw-None-09600)
```
Bad news: I haven't found the solution

Good news: It seems to make no difference

**Under Construction...**