# EAST
## 问题
1. 生成gt时，shrunk quadrange是为啥？  
   解答：
   所有基于分割的检测方法都有一个通病就是很难把相邻的两个文本框分开，所以所有基于分割的方法都要解决这个问题。而east解决这个问题的方法就是只将文本框的中心认为属于文本框，然后回归中心到四个边框的距离即可，对于边上的其实有时候很难识别的足够准确。所以就不对他们进行回归了
2. 不同的形状对应不同的输出，训练时是每个样本都做了两种训练？两个模型？使用时人为选择模型？  
   解答：这是作者给出的两种训练策略，quad比rbox检测框对倾斜文本包围更加精确一些
## 关键点
1. 提出了基于two-stage的文本检测方法：全卷积网络(FCN)和非极大值抑制(NMS)，消除中间过程冗余，减少检测时间．
2. 该方法即可以检测单词级别，又可以检测文本行级别．检测的形状可以为任意形状的四边形：即可以是旋转矩形(下图中绿色的框)，也可以是普通四边形(下图中蓝色的框)）．
3. 采用了Locality-Aware NMS来对生成的几何进行过滤
4. 该方法在精度和速度方面都有一定的提升．
## 网络结构
![sparkles](pipelines_of_EAST.png)
只有两个stage: 多通道全卷积和LNMS（局部感知NMS）  
具体流程：  
![sparkles](structure_of_EAST.png)  
1. 先用通用网络作为base net提取特征（论文使用的是Pvanet,也可以采用VGG16， Resnet等）
2. 基于上述主干特征提取网络，抽取不同level的feature map（尺寸分别是输入图像的 1/32, 1/16, 1/8, 1/4），利用不同尺度的特征图，解决文本行尺度变换剧烈的问题，early stage可用于预测小的文本行，late stage可用于预测大的文本行
3. 特征合并层，将抽取的特征进行merge，采用U-net的方法
4. 网络输出层，包含文本得分（one channel）和文本形状(multi channel)。不同文本形状（RBOX和QUAD）对应的输出不同。

## 训练
### ground truth  
RBOX（Rotation box）的Geometry map表示为四个通道的同轴边界框R和一个通道的旋转角，四个通道代表像素点与矩形框四条边的距离。 
QUAD（Quadrange, 四边形）的Geometry map表示为8个相对于角点的偏移。   
![sparkles](label_of_EAST.png)  
黄色虚线是文本框，绿色是缩小后的文本框。图b是得分图，图c中，粉色框为黄色虚线框的最小外接矩，生成RBOX的geometry map即白色框中每个点到粉框四条边的距离，因此生成d中四个channel的map，旋转角度e即粉框的水平倾斜角，所有像素点相同，1个channel。  
对于QUAD，则是计算白框每个像素点到黄色四边形框的四个角点的偏移，每个点有x和y两个坐标，因此是8个通道。  

### Loss
#### Score Loss
![sparkles](score_loss_of_EAST.png)  
#### Geometry loss
##### RBOX:
![sparkles](RBOX_loss_of_EAST.png)  
##### QUAD:
![sparkles](QUAD_loss_of_EAST.png) 
![sparkles](QUAD_loss2_of_EAST.png)  

## 相关概念
### 标准NMS：  
1. 将所有检出的output bbox按cls score划分(如文本检测仅包含文1类，即将output bbox按照其对应的cls score划分为2个集合，1个为bg类，bg类不需要做NMS而已) 
2. 在每个集合内根据各个bbox的cls score做降序排列，得到一个降序的list_k 
3. 从list_k中top1 cls score开始，计算该bbox_x与list中其他bbox_y的IoU，若IoU大于阈值T，则剔除该bbox_y，最终保留bbox_x，从list_k中取出 
4. 对剩余的bbox_x，重复step-3中的迭代操作，直至list_k中所有bbox都完成筛选； 
5. 对每个集合的list_k，重复step-3、4中的迭代操作，直至所有list_k都完成筛选；  
### LNMS：  
1. 先对所有的output box集合结合相应的阈值（大于阈值则进行合并，小于阈值则不和并），依次遍历进行加权合并，得到合并后的bbox集合； 
2. 对合并后的bbox集合进行标准的NMS操作

