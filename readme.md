# 创建专用环境
conda create -n dl-learning python=3.10
conda activate dl-learning

# 安装PyTorch套件
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# RNN
![img_1.png](images/img_1.png)
![img.png](images/img.png)

## RNN在时间维度上共享权重矩阵,即 图示W、U、V

# LSTM

![img_2.png](images/img_2.png)
![img_3.png](images/img_3.png)
![img_4.png](images/img_4.png)
![img_5.png](images/img_5.png)

# transformer
### 基本架构
![img_6.png](images/img_6.png)

