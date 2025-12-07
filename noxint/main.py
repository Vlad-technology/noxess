import tensorflow as tf
import matplotlib.pyplot as plt

def iii():
    inp = input()
    inp_index = [int(char) for char in inp]
    return inp_index

#defs neuronos:

def n1():
    inp_tensor = tf.constant(inp_index, dtype=tf.float32)
    n1_1 = tf.sigmoid(inp_tensor)
    n1_2 = tf.math.tanh(inp_tensor)
    n1_3 = tf.nn.relu(inp_tensor)
    inp_index_1 = [n1_1.numpy()[0], n1_2.numpy()[0], n1_3.numpy()[0]]
    return inp_index_1

def n2():
    inp_tensor = tf.constant(inp_index_1, dtype=tf.float32)
    n2_1 = tf.sigmoid(inp_tensor)
    n2_2 = tf.math.tanh(inp_tensor)
    n2_3 = tf.nn.relu(inp_tensor)
    inp_index_2 = [n2_1.numpy()[0], n2_2.numpy()[0], n2_3.numpy()[0]]
    return inp_index_2


#main logic

print("pls input your number")

inp_index = iii()
inp_index_1 = n1()
inp_index_2 = n2()

x = [0, 0, 0]
y = [inp_index_2[0], inp_index_2[1], inp_index_2[2]]

print(inp_index_2[0])
print(inp_index_2[1])
print(inp_index_2[2])

plt.plot(x, y, marker='o')
plt.title("graph")
plt.xlabel("")
plt.ylabel("")
plt.grid()
plt.show()