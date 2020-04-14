# Mask TextSpotter
## 关键点
- 检测、识别端到端
- 语义分割适用于任意形状文本行

## 网络结构
![sparkles](Structure_of_Mask_TextSpotter.jpg)
网络结构包含四个部分：1.FPN作为backbone；2.RPN生成候选区域；3.Fast RCNN进行边界回归；4.mask 分支进行文本行分割、字符分割和文本行识别  
