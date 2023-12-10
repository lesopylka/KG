from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LightSource
from numpy import sin, cos
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

tempx = 0
tempy = 0
tempz = 0
phiLight = 0
psiLight = 0
alphaT = 0.5
normal = []
flags = []
ResX = []
ResY = []
ResZ = []
point_of_view = [5,0,0]
key = 0

def onButtonAddClicked(event):
    global key
    global phiLight
    global psiLight
    if key == 0:
        key = 1
    else:
        key = 0
    drawWithLight(phiLight, psiLight)


def linear_sub(a, b):
    if (a[0] * b[0] - a[1] * b[1] + a[2] * b[2] > 0):
        return 1
    else:
        return 0

def det_of_mtx(a21,a22,a23,a31,a32,a33):
    return [a22 * a33 - a23 * a32, a21*a33 - a23*a31, a21 * a32 - a22 * a31]

def alphaChanges(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global phiLight
    global psiLight
    global alphaT
    global key
    alphaT = value
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 1:
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=value, lightsource=ls)
    else:
        ax.plot_wireframe(x, y, z, color="black", alpha=alphaT)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def getNormals():
    global x
    global y
    global z
    global ax
    global fig
    global normal
    #flags = []
    normal = []
    points = []
    res = []
    print(len(x[0]) - 1)
    for idx in range(len(x[0]) - 1):
        temp = [x[0][0], y[0][0], z[0][0]]
        temp1 = [x[1][idx], y[1][idx], z[1][idx]]
        temp2 = [x[1][idx + 1], y[1][idx + 1], z[1][idx + 1]]
        res.append(temp)
        res.append(temp1)
        res.append(temp2)
        points.append(np.array(res[0]) - np.array(res[1]))
        points.append(np.array(res[0]) - np.array(res[2]))
        res = []
    for i in range(len(x) - 1):
        for j in range(len(x[0]) - 1):
            temp = [x[i][j], y[i][j], z[i][j]]
            temp1 = [x[i+1][j], y[i+1][j], z[i+1][j]]
            temp2 = [x[i][j+1], y[i][j+1], z[i][j+1]]
            res.append(temp)
            res.append(temp1)
            res.append(temp2)
            points.append(np.array(res[0]) - np.array(res[1]))
            points.append(np.array(res[0]) - np.array(res[2]))
            res = []
    for idx2 in range(len(x[0])):
        temp = [x[len(x[0]) - 1][len(x[0]) - 1], y[len(x[0]) - 1][len(x[0]) - 1], z[len(x[0]) - 1][len(x[0]) - 1]]
        temp1 = [x[len(x[0]) - 2][idx2 - 1], y[len(x[0]) - 2][idx2 - 1], z[len(x[0]) - 2][idx2 - 1]]
        temp2 = [x[len(x[0]) - 2][idx2], y[len(x[0]) - 2][idx2], z[len(x[0]) - 2][idx2]]
        res.append(temp)
        res.append(temp1)
        res.append(temp2)
        points.append(np.array(res[0]) - np.array(res[1]))
        points.append(np.array(res[0]) - np.array(res[2]))
        res = []
    print(len(points) - 1)
    for i in range(0, len(points) - 1, 2):
        normal.append(det_of_mtx(points[i][0], points[i][1], points[i][2], points[i+1][0], points[i+1][1], points[i+1][2]))
    for i2 in range(len(normal)):
        flags.append(linear_sub(normal[i2], point_of_view))
    print(flags)

def visible_sides():
    global x
    global y
    global z
    global ax
    global fig
    global flags
    global alphaT
    global key
    global ResX
    global ResY
    global ResZ
    ResX = []
    ResY = []
    ResZ = []
    x2 = []
    y2 = []
    z2 = []
    x3 = []
    y3 = []
    z3 = []
    x4 = []
    y4 = []
    z4 = []
    tmp = 0
    k = 0
    for temp in range(len(x[0])):
        if (flags[temp] == 1):
            x2.append(x[0][0])
            y2.append(y[0][0])
            z2.append(z[0][0])
            ResX.append([x[0][0], y[0][0], z[0][0]])
            x2.append(x[1][temp])
            y2.append(y[1][temp])
            y2.append(y[1][temp])
            ResX.append([x[1][temp], y[1][temp], z[1][temp]])
            x2.append(x[1][temp + 1])
            y2.append(y[1][temp + 1])
            z2.append(z[1][temp + 1])
            ResX.append([x[1][temp + 1], y[1][temp + 1], z[1][temp + 1]])
    tmp = temp + 1
    print(ResX)
    for i in range(1, len(x) - 1):
        for j in range(1, len(x[0]) - 1):
            if (flags[tmp] == 1):
                ResX.append([x[i][j], y[i][j], z[i][j]])
                ResX.append([x[i + 1][j], y[i + 1][j], z[i + 1][j]])
                ResX.append([x[i][j + 1], y[i][j + 1], z[i][j+1]])
                ResX.append([x[i + 1][j + 1], y[i + 1][j + 1], z[i + 1][j + 1]])
            tmp += 1
        x3 = []
        y3 = []
        z3 = []

    for i in range(1, len(x[0])):
        if (tmp < len(flags)) and (flags[tmp] == 1):
            ResX.append([x[len(x) - 1][len(x[i]) - 1], y[len(x)][len(x[i]) - 1], z[len(x)][len(x[i]) - 1]])
            ResX.append([x[i - 1][len(x[i]) - 2], y[i - 1][len(x[i]) - 2], z[i - 1][len(x[i]) - 2]])
            ResX.append([x[i][len(x[i]) - 2], y[i][len(x[i]) - 2], z[i][len(x[i]) - 2]])
        tmp += 1
    print(ResX)
    print(len(ResX[0]))
    max_resx = 0
    max_resy = 0
    max_resz = 0
    for i in range(len(ResX)):
        ResX[i] = list(dict.fromkeys(ResX[i]))
        if max_resx < len(ResX[i]):
            max_resx = len(ResX[i])
    for i in range(len(ResY)):
        ResY[i] = list(dict.fromkeys(ResY[i]))
        if max_resy < len(ResY[i]):
            max_resy = len(ResY[i])
    for i in range(len(ResZ)):
        ResZ[i] = list(dict.fromkeys(ResZ[i]))
        if max_resz < len(ResZ[i]):
            max_resz = len(ResZ[i])
    for i in range(len(ResX)):
        print(len(ResX[i]))
    ResXT = []

    print(max_resx, max_resy, max_resz)
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    ResX3 = np.vstack((ResX[1:len(ResX) - 1]))
    ResY3 = np.vstack((ResY[1:len(ResX) - 1]))
    ResZ3 = np.vstack((ResZ[1:len(ResX) - 1]))
    if key == 0:
        ax.pol
    else:
        ax.plot_surface(x, y, z, color="black", alpha=alphaT, lightsource= ls)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def drawWithLight(phiLight, psiLight):
    global x
    global y
    global z
    global ax
    global fig
    global key
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 0:
        ax.plot_wireframe(x, y, z, color="black", alpha=alphaT)
    else:
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=alphaT, lightsource= ls)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def lightPhiChange(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global phiLight
    global psiLight
    global normal
    global flags
    global alphaT
    global key
    phiLight = value
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 0:
        ax.plot_wireframe(x, y, z, color="black", alpha=alphaT)
    else:
        print(x[0])
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=alphaT, lightsource= ls)
    getNormals()
   # visible_sides()
    flags = []
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    #drawWithLight(xLight, yLight, zLight)

def lightPsiChange(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global phiLight
    global psiLight
    psiLight = value
    drawWithLight(phiLight, psiLight)


def precisionRotateX(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempx
    for idx in range(value):
        for i in range(len(x)):
            for j in range(len(x[0])):
                y[i][j] = y[i][j] * cos(np.radians(1)) + (-1) * z[i][j] * sin(np.radians(1))
                z[i][j] = y[i][j] * sin(np.radians(1)) + z[i][j] * cos(np.radians(1))

def precisionRotateY(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempy
    for idx in range(value):
        for i in range(len(y)):
            for j in range(len(y[0])):
                x[i][j] = x[i][j] * cos(np.radians(1)) + z[i][j] * sin(np.radians(1))
                z[i][j] = (-1) * x[i][j] * sin(np.radians(1)) + z[i][j] * cos(np.radians(1))

def precisionRotateZ(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempz
    for idx in range(value):
        for i in range(len(y)):
            for j in range(len(y[0])):
                x[i][j] = x[i][j] * cos(np.radians(1)) + (-1) * y[i][j] * sin(np.radians(1))
                y[i][j] = x[i][j] * sin(np.radians(1)) + y[i][j] * cos(np.radians(1))

def reDraw(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempx
    global tempy
    global tempz
    global alphaT
    global key
    global phiLight
    global psiLight
    x, y = np.mgrid[-2:2:complex(value), -2:2:complex(value)]

    z = np.sqrt(4. * (x ** 2 + y ** 2) / 1. + 1)

    xcolors = z - min(z.flat)
    xcolors = xcolors / max(xcolors.flat)
    # alpha controls opacity
    ax.plot_wireframe(x, y, z, color="black", alpha=0.5)
    precisionRotateX(tempx)
    precisionRotateY(tempy)
    precisionRotateZ(tempz)
    # alpha controls opacity
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 1:
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=alphaT, lightsource=ls)
    else:
        ax.plot_wireframe(x, y, z, color= 'black', alpha= alphaT)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def rotateX(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempx
    global alphaT
    global key
    global phiLight
    global psiLight
    value_temp = tempx
    tempx = value
    value = value - value_temp

    for i in range(len(y)):
        for j in range(len(y[0])):
            y[i][j] = y[i][j]*cos(np.radians(value)) + (-1) * z[i][j] * sin(np.radians(value))
            z[i][j] = y[i][j]*sin(np.radians(value)) + z[i][j] * cos(np.radians(value))
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 0:
        ax.plot_wireframe(x, y, z, color="black", alpha=alphaT)
    else:
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=alphaT, lightsource= ls)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def rotateY(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempy
    global alphaT
    global key
    global phiLight
    global psiLight
    value_temp = tempy
    tempy = value
    value = value - value_temp
    for i in range(len(y)):
        for j in range(len(y[0])):
            x[i][j] = x[i][j]*cos(np.radians(value)) + z[i][j] * sin(np.radians(value))
            z[i][j] = (-1) * x[i][j]*sin(np.radians(value)) + z[i][j] * cos(np.radians(value))
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 0:
        ax.plot_wireframe(x, y, z, color="black", alpha=alphaT)
    else:
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=alphaT, lightsource= ls)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def rotateZ(value: np.float64):
    global x
    global y
    global z
    global ax
    global fig
    global tempz
    global alphaT
    global key
    global phiLight
    global psiLight
    value_temp = tempz
    tempz = value
    value = value - value_temp
    for i in range(len(y)):
        for j in range(len(y[0])):
            x[i][j] = x[i][j]*cos(np.radians(value)) + (-1) * y[i][j] * sin(np.radians(value))
            y[i][j] = x[i][j]*sin(np.radians(value)) + y[i][j] * cos(np.radians(value))
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection='3d')
    ls = LightSource(phiLight, psiLight)
    if key == 0:
        ax.plot_wireframe(x, y, z, color="black", alpha=alphaT)
    else:
        ax.plot_surface(x, y, z, color=plt.cm.Blues(0.2), alpha=alphaT, lightsource= ls)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


fig = plt.figure()
fig.subplots_adjust(left=0.07, right=0.7, top=0.95, bottom=0.4)


axes_slider_X = plt.axes([0.05, 0.25, 0.7, 0.04])
slider_X = Slider(axes_slider_X,
                    label='X',
                    valmin=-180,
                    valmax=180,
                    valinit=0,
                    valstep=1
                  )
axes_slider_Y = plt.axes([0.05, 0.19, 0.7, 0.04])
slider_Y = Slider(axes_slider_Y,
                    label='Y',
                    valmin=-180,
                    valmax=180,
                    valinit=1,
                    valstep=1
                  )
axes_slider_Z = plt.axes([0.05, 0.13, 0.7, 0.04])
slider_Z = Slider(axes_slider_Z,
                    label='Z',
                    valmin=-180,
                    valmax=180,
                    valinit=1,
                    valstep=1
                  )

axes_slider_Precision = plt.axes([0.2, 0.07, 0.5, 0.04])
slider_Precision = Slider(axes_slider_Precision,
                    label='Аппроксимация',
                    valmin=2,
                    valmax=30,
                    valinit=15,
                    valstep=1
                  )


axes_slider_PhiLight = plt.axes([0.7, 0.35, 0.03, 0.5])
slider_Phi = Slider(axes_slider_PhiLight,
                    label='PHI',
                    valmin=-180,
                    valmax=180,
                    valinit=0,
                    valstep=1,
                    orientation='vertical'
                  )
axes_slider_PsiLight = plt.axes([0.75, 0.35, 0.03, 0.5])
slider_Psi = Slider(axes_slider_PsiLight,
                    label='PSI',
                    valmin=-180,
                    valmax=180,
                    valinit=0,
                    valstep=1,
                    orientation='vertical'
                  )

axes_slider_Transparent = plt.axes([0.8, 0.35, 0.03, 0.5])
slider_Tr = Slider(axes_slider_Transparent,
                    label='Tr',
                    valmin=0,
                    valmax=1,
                    valinit=0.5,
                    valstep=0.05,
                    orientation='vertical'
                  )

axes_button_add = plt.axes([0.8, 0.05,0.15,0.05])
button_add = Button(axes_button_add, 'Заполнить')

ax = fig.add_subplot(111, projection='3d')

x,y = np.mgrid[-2:2:15j, -2:2:15j]

z = np.sqrt(4.*(x**2 + y**2)/1. + 1)

xcolors = z - min(z.flat)
xcolors = xcolors/max(xcolors.flat)
# alpha controls opacity
ax.plot_wireframe(x, y, z, color="black", alpha=0.5)


ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

slider_X.on_changed(rotateX)
slider_Y.on_changed(rotateY)
slider_Z.on_changed(rotateZ)
slider_Precision.on_changed(reDraw)
slider_Phi.on_changed(lightPhiChange)
slider_Psi.on_changed(lightPsiChange)
slider_Tr.on_changed(alphaChanges)
button_add.on_clicked(onButtonAddClicked)

plt.show()