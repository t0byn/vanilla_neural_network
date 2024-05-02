A simple two layer neural network for recognizing handwritten digits written in C++. The neural network setting (the number of epoch, the number of mini-batch, the number of neuron each layer etc.) basically copy from this blog post (https://www.ea.com/seed/news/machine-learning-game-devs-part-3).
It took about 10 minute to finish training on my computer. This neural network achieved roughly 95% accuracy on the testing data set.

If you have visual studio installed, open `Developer Command Prompt for VS` and use command `cl.exe main.cc /link /out:nn.exe` to compile. Or use cmake to compile. I haven't test on other compiler or platform. 

reference:
- https://www.ea.com/seed/news/machine-learning-game-devs-part-1
- https://www.ea.com/seed/news/machine-learning-game-devs-part-2
- https://www.ea.com/seed/news/machine-learning-game-devs-part-3
- http://yann.lecun.com/exdb/mnist/