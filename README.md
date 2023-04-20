# Animated character voice quad classifier 动画人物语音四分类器

# Abstract-in chinese
* 分类器所用的神经网络模型来自于自己训练。
* 模型的网络架构：基于Parallel is all you want 网络架构，在原始网络架构中先两个CNN的输出张量，经过selfattention层后，再与transfomer层输出拼接，一并输入到最后的全连接层。
* 模型数据集：自建数据集。数据集音频来源辛普森一家S33E01+S33E02，有337条语音；由于版权，暂不分享。
* |--------------------------------------------------------------------------------------|
* 结合我在分类器中写的model文件以及Parallel is all you want原github网址，相信你可以替换数据集，并训练出自己的分类器！
* 可联系18810967669@qq.com与我进行学习交流。（ps：我自己还不会上传，麻烦同学上传，所以github上可能联系不到我。
* |--------------------------------------------------------------------------------------|
* 分类器有两种可视化。一是，`predict_emo_in_file.py`文件，运行后可在命令行进行交互；二是the_web文件夹下的文件，`test_web.py`点击后会输出网址，点击进入浏览器即可上传音频文件。
* 分类器均支持`wav\mp3`两种文件格式。`predict_emo_in_file.py`文件支持输入文件夹批量预测，并保存预测结果；`test_web.py`网页分类器，支持单个音频文件上传。
* 欢迎来玩~~

# the cite in my classfier 实验基础
|Baseline:[See Notebook for Code and Explanations](https://nbviewer.jupyter.org/github/IliaZenkov/transformer_cnn_parallel_audio_classification/blob/main/notebooks/Parallel_is_All_You_Want.ipynb)
@misc{Zenkov-Transformer-CNN-SER,
  author = {Zenkov, Ilia},
  title = {transformer-cnn-emotion-recognition},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/QiaFan/transformer-cnn-emotion-recognition}},
}
|selfattention:- Pytorch implementation of ["Attention Is All You Need---NIPS2017"](https://arxiv.org/pdf/1706.03762.pdf)
(ps:资料从FightingCV获得，找不到引用，在此推荐他们的公众号和github
作为[**FightingCV公众号**](https://mp.weixin.qq.com/s/m9RiivbbDPdjABsTd6q8FA)和 **[FightingCV-Paper-Reading](https://github.com/xmu-xiaoma666/FightingCV-Paper-Reading)** 的补充，本项目的宗旨是从代码角度，实现🚀**让世界上没有难读的论文**🚀。
-->

|  49% Accuracy     | my modle in test of my datasets|
|---------------------------|------------------|

## Cite
If you find this work useful in your own research, please cite as follows:

```
@misc{anima_SER_quar_classfier,
  author = {buliandan},
  title = {Animated character voice quad classifier},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/buliandan/anima_SER_quar_classfier}},
}
```
