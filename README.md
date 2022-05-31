# 百度网盘AI大赛-图像处理挑战赛:手写文字擦除第1名方案，水印智能消除赛第1名方案
百度网盘AI大赛-图像处理挑战赛:手写文字擦除A榜第2名，B榜第3方案    
比赛连接：[百度网盘AI大赛：手写文字擦除(赛题二)](https://aistudio.baidu.com/aistudio/competition/detail/129/0/introduction)
比赛连接：[百度网盘AI大赛-图像处理挑战赛：水印智能消除赛)](https://aistudio.baidu.com/aistudio/competition/detail/209/0/introduction)

## 一、赛题背景
对比赛给定的带有手写痕迹的试卷图片进行处理，擦除相关的笔，还原图片原本的样子
![](https://ai-studio-static-online.cdn.bcebos.com/af2816877d054080987de1f47679fa656e5f498fd39744f5a9f94cc6c5a4fb9d)

## 二、数据分析
**数据划分**：使用1000张做为训练集，81张作为验证集。    
官方提供了训练集1081对，测试集A、B各200张。包含以下几个特征：    
1.图像分辨率普遍较大    
2.手写字包含红黑蓝多种颜色，印刷字基本为黑色    
3.手写字除了正常文字外，还包含手画的线段、图案等内容    
4.试卷上的污渍、脏点也属于需要去除的内容    
5.手写字和印刷字存在重叠    

**mask**：根据原始图片和标签图像的差值来生成mask数据    
计算RGB通道的平均差值    
平均差值在20以上的设为 1    
平均差值在20以下的设为 差值/20    

![](https://ai-studio-static-online.cdn.bcebos.com/255b0b9dd6e8426fae2d9f01c6bd17229fd4dbb37a5741539ba8d8ea87fd10f3)

## 三、模型设计
网络模型，是基于开源的EraseNet，然后整体改成了Paddle版本。同时也尝试了最新的PERT：一种基于区域的迭代场景文字擦除网络。基于对比实验，发现ErastNet，在本批次数据集上效果更好。从网络结构图上可以直观的看出ErastNet是多分支以及多阶段网络其中包括mask生成分支和两阶段图像生成分支。此外整个网络也都是基于多尺度结构。在损失函数上，原版的ErastNet使用了感知损失以及GAN损失。两个损失函数，是为了生成更加逼真的背景。但是本赛题任务的背景都是纯白，这两个损失是不需要的，可以直接去除。此外，由于ErastNet网络是由多尺度网络组成，结合去摩尔纹比赛的经验，我把ErastNet网络的Refinement替换成了去摩尔纹比赛使用的多尺度网络
双模型融合：    
模型一：erasenet去掉判别器部分，仅保留生成器    
![](https://ai-studio-static-online.cdn.bcebos.com/7546d26870a44fce9b5f118b8fc8e8501b7f4ed1e807468ebece4c9d21209ac0)
模型二：erasenet二阶段网络使用基于Non-Local的深度编解码结构    
![](https://ai-studio-static-online.cdn.bcebos.com/67f2b22dca8a491cad844354f2ba81601190f4bda4e44524a115b8c715bedbfb)

## 四、训练细节

**训练数据：**    
增强仅使用横向翻转和小角度旋转，保留文字的先验    
随机crop成512x512的patch进行训练    
    
**训练分为两阶段：**    
第一阶段损失函数为dice_loss + l1 loss    
第二阶段损失函数只保留l1 loss    

## 五、测试细节

测试trick：    
**分块测试**，把图像切分为512x512的小块进行预测，保持和训练一致    
**交错分块测试**，测试图像增加镜像padding，且分块时边缘包含重复部分，每次预测仅保留每块预测结果的中心部分，这么做的原因是图像边缘信息较少，预测效果要差于中心部分
测试时对**测试**数据使用了横向的镜像**增强**    
测试时将两个**模型**的预测结果进行**融合**    

## 六、上分策略

![](https://ai-studio-static-online.cdn.bcebos.com/88dd53709c1f47aca80f9ce63e344e8494c44c59b9534367b7aa4b5b0034caad)

## 七、其他
data：定义数据加载      
loss：定义损失函数       
model：定义网络模型    
compute_mask.py：生成mask文件    
test.py: 测试脚本    
train.py: 训练脚本    

代码运行:    
1.指定数据文件夹    
2.运行sh train.sh 生成mask并开始训练    
3.指定测试文件夹和模型路径，执行sh test.sh开始测试    
## 预训练模型    
https://aistudio.baidu.com/aistudio/projectdetail/3439691    
运行项目，下载预训练模型，同时可以进行在线测试。
