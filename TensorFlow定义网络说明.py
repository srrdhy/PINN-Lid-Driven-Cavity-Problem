import tensorflow as tf

class PINN(tf.keras.Model):   #  x y -> u v p
    def __init__(self, num_hidden_layers=8, num_neurons_per_layer=20, **kwargs):
        super(PINN, self).__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.hidden_layers = [] # 用于存储创建的隐藏层

        for _ in range(self.num_hidden_layers): # 对于隐藏层的每一层
            layer = tf.kears.layers.Dense( # 每一层是一个 Dense（全连接）层
                self.num_neurons_per_layer, # 每个隐藏层的神经元数量由 num_neurons_per_layer 确定
                activation = 'tanh', # 激活函数
                kernel_initializer='glorot_normal', # 使用 Glorot 正态分布初始化权重
            )
            self.hidden_layers.append(layer)

        # 定义了三个输出层：u_output、v_output 和 p_output，每个输出层都有一个神经元，表示模型的输出
        self.u_output = tf.keras.layers.Dense(1, activation = None)  # activation=None 表示输出层没有激活函数
        self.v_output = tf.keras.layers.Dense(1, activation = None)
        self.p_output = tf.keras.layers.Dense(1, activation = None)

    def call(self, inputs): # call 方法是模型的前向传播函数  inputs 是模型的输入 形状为 (batch_size, 2)
        x, y = inputs[:, 0:1], inputs[:, 1:2]  # 分别提取输入中的 x 和 y 坐标  形状是 (batch_size, 1)
        X = tf.concat([x, y], axis=1)  # 将 x 和 y 在列上拼接，形成一个形状为 (batch_size, 2) 的张量
        for layer in self.hidden_layers:  # 输入 X 会通过每一层隐藏层进行处理
            X = layer(X)   
        u = self.u_output(X)  # u, v, p 分别是通过三个输出层得到的模型预测结果
        v = self.v_output(X)
        p = self.p_output(X)
        return u, v, p
    
    def get_config(self):  # 返回模型的配置字典，用于序列化模型时记录模型的参数（如层数和每层神经元数量）
        config = super(PINN, self).get_config()  # 获取父类（tf.keras.Model）的配置
        config.update({
            'num_hidden_layers': self.num_hidden_layers,
            'num_neurons_per_layer': self.num_neurons_per_layer
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)  # 通过 cls(**config) 将配置字典中的参数传递给模型的构造函数来重新创建模型
