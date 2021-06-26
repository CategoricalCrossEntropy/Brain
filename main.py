import pygame
pygame.init()

import numpy as np

import matplotlib.pyplot as plt


class Neuron:
    def __init__(self):
        self.balance = 0.7
        self.power = np.random.random()
        self.limit = 0.6
        self.braking_speed = 0.2
        self.acceleration_speed = 0.1
        self.input_signal = np.zeros((2, 2))
        self.weights = np.random.rand(2, 2) * 3
        self.lr = 10
        self.History = []

    def eval_input_signal(self):
        self.power += np.sum(self.input_signal * self.weights)
        reward = self.lr * (self.power - self.balance)
        delta_w = (self.input_signal * self.weights) * reward
        self.weights += delta_w
        self.weights = self.weights.clip(0, 1)

    def step(self):
        delta = self.power - self.balance
        if delta > 0:
            self.power -= delta * self.braking_speed
        else:
            self.power -= delta * self.acceleration_speed
            
        self.eval_input_signal()    
        output_signal = max(self.power - self.limit, 0)
        if output_signal > 0:
            output_signal = self.power / 4
            self.power = 0
        self.History.append(self.power)
        return output_signal


    
def color(val):
    if val < 0:
        val = 0
    if val > 1:
        val = 1
    return tuple([int(val * 255) for _ in range(3)])


def put_input(neuron, Signals, i, j):
    if j == 0:
        neuron.input_signal[0][0] = 0
    else:
        neuron.input_signal[0][0] = Signals[i][j - 1]

    if i == 0:
        neuron.input_signal[0][1] = 0
    else:
        neuron.input_signal[0][1] = Signals[i - 1][j]

    if i == Signals.shape[0] - 1:
        neuron.input_signal[1][0] = 0
    else:
        neuron.input_signal[1][0] = Signals[i + 1][j]

    if j == Signals.shape[1] - 1:
        neuron.input_signal[1][1] = 0
    else:
        neuron.input_signal[1][1] = Signals[i][j + 1]

def act(Network):
    if np.random.random() < 0.1:
        Network[1][1].power += 100
    for i in range(len(Network)):
        for j in range(len(Network[i])):
            neuron = Network[i][j]
            put_input(neuron, Signals, i, j)
    for i in range(len(Network)):
        for j in range(len(Network[i])):
            neuron = Network[i][j]
            Signals[i][j] = neuron.step()
            rect = (j * size + ofs_x, i * size + ofs_y,
                    size, size)
            pygame.draw.rect(sc, color(neuron.power), rect)

    if step % 100 == -1:
        for x in range(3):
            plt.plot(Network[x + 9][12].History)
        plt.show()
        print(Network[12][12].weights)
            
FPS = 20000
clock = pygame.time.Clock()
wid, hght = 800, 200
ofs_x, ofs_y = 0, 0
size = 10
sc = pygame.display.set_mode((wid, hght))

Network = [[Neuron() for x in range(ofs_x // size, (wid - ofs_x) // size)]
                     for y in range(ofs_y // size, (hght - ofs_y) // size)]
Signals = np.zeros((len(Network), len(Network[0])))

for step in range(100000):
    sc.fill(pygame.Color('black'))
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            pygame.quit()
            
    act(Network)
                
    clock.tick(FPS)
    pygame.display.update()
