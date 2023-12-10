import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

#Определение параметров
#a=b
a = 0.06
c = 0.035
n = 19

#Функция для создания вершин
def create_hyperboloid(a, c, n):
    vertices = []
    for i in range(n):
        t = i * 2 * math.pi / n
        t_next = (i + 1) * 2 * math.pi / n
        cos_t = math.cos(t)
        cos_t_next = math.cos(t_next)
        sin_t = math.sin(t)
        sin_t_next = math.sin(t_next)

        for j in range(n):
            u = j * math.pi / n
            u_next = (j + 1) * math.pi / n
            ch_u = math.cosh(u)
            ch_u_next = math.cosh(u_next)
            sh_u = math.sinh(u)
            sh_u_next = math.sinh(u_next)

            x = [a * sh_u * cos_t, a * sh_u_next * cos_t, a * sh_u_next * cos_t_next, a * sh_u * cos_t_next]
            y = [a * ch_u * sin_t, a * ch_u_next * sin_t, a * ch_u_next * sin_t_next, a * ch_u * sin_t_next]
            z = [c * ch_u, c * ch_u_next, c * ch_u_next, c * ch_u]

            vertices.append([(x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2])])
            vertices.append([(x[0], y[0], z[0]), (x[2], y[2], z[2]), (x[3], y[3], z[3])])

    return vertices

vertices = create_hyperboloid(a, c, n)

#Инициализация Pygame и OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL) #доступ к дисплею, размер, позиция

gluPerspective(45, (display[0] / display[1]), 0.1, 50.0) #настраивает матрицу проекции перспективы
glTranslatef(0.0, 0.0, -5) #умножает текущую матрицу на матрицу преобразования
# позволяют включить возможности OpenGL
glEnable(GL_LIGHTING) #параметры освещения для вычисления цвета или индекса вершины
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_DEPTH_TEST) #сравнение глубины
glLightfv(GL_LIGHT0, GL_POSITION, [2, 3, 1, 3]) #значения параметров источника света
glMaterial(GL_FRONT, GL_DIFFUSE, [1.0, 0.0, 0.0, 1.0]) #параметры материала для модели освещения

#Основной цикл отображения
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBegin(GL_TRIANGLES) #разделяет вершины примитива, рассматривает каждый триплет вершин как независимый треугольник
    for triangle in vertices:
        for vertex in triangle:
            glVertex3fv(vertex)
    glEnd()

    pygame.display.flip() #обновит содержимое всего дисплея
    pygame.time.wait(10)
