'''
f''(x) = 2m/hbar^2(V(x)- E)f(x)
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import math
import sys

sys.setrecursionlimit(100000)

omega0 = 1
m = 1
even = True
H = 12
Pot = 'Well'

def V(x):
    global Pot
    if Pot == 'Well':
        return VWell(x)
    elif Pot == 'Osc':
        return VOsc(x)
    else:
        V1 = 1
        V2 = 1
        k1 = 1 + np.sqrt(5)
        k2 = 2
        if x > -2 and x < -2:
            return V1*np.sin(k1*x) + V2*np.sin(k2*x)
        else:
            return math.inf

def VWell(x):
    global H
    if x<-2:
        return H
    elif x > 2:
        return H
    else:
        return 0

def VOsc(x):
    return 1/2 * omega0 * m * x**2

def VAbs(x):
    return H*abs(x)

def EulerAndPlot():
    global PhysStop, e0, E
    xtab = np.linspace(0, PhysStop*3, 1000)
    f = np.zeros(1000)
    delta = xtab[1]-xtab[0]
    if even:
        f[0] = 2
        df = 0
    else:
        f[0] = 0
        df = 7

    for i in range(len(xtab)-1):
        df = df + delta*e0*(V(xtab[i]) - E)*f[i]
        f[i+1] = f[i] + delta*df
    return xtab, f



PhysStop = 2
e0 = np.pi**2/16
E = H/2


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
xtab, f = EulerAndPlot()
l, = plt.plot(xtab, f, lw=2, color='red')
if even:
    l2, = plt.plot(-xtab, f, lw=2, color='red')
else:
    l2, = plt.plot(-xtab, -f, lw=2, color='red')
lV, = plt.plot(xtab, [V(x) for x in xtab], color = 'orange')
lV2, = plt.plot(-xtab, [V(x) for x in -xtab], color = 'orange')
lE, =  plt.plot(-xtab, [E]*len(xtab), color = 'blue')
lE2, = plt.plot(xtab, [E]*len(xtab), color = 'blue')
plt.axis([-4*PhysStop, 4*PhysStop, -10, 20])
vl = plt.axvline(x = PhysStop)
vl2 = plt.axvline(x = -PhysStop)

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sE = Slider(axfreq, 'E', 0, 30.0, valinit=E, valstep=0.001)
sH = Slider(axamp, 'H', 5, 20.0, valinit=H)

fprev = f

def update(val):
    global E, H, PhysStop
    #amp = samp.val
    E = sE.val
    H = sH.val
    lE.set_ydata([E]*len(xtab))
    lE2.set_ydata([E]*len(xtab))
    _, f = EulerAndPlot()
    l.set_ydata(f)
    if even:
        l2.set_ydata(f)
    else:
        l2.set_ydata(-f)
    lV.set_ydata([V(x) for x in xtab])
    lV2.set_ydata([V(x) for x in -xtab])
    try:
        idx = np.argwhere(np.diff(np.sign(np.array([E for _ in xtab]) - np.array([V(x) for x in xtab])))).flatten()
        if idx.size > 0:
            vl.set_xdata(xtab[idx])
            PhysStop = xtab[idx]
        idx = np.argwhere(np.diff(np.sign(np.array([E for _ in -xtab]) - np.array([V(x) for x in -xtab])))).flatten()
        if idx.size > 0:
            vl2.set_xdata(-xtab[idx])
        pass
    except ValueError:
        PhysStop = 10
        vl.set_xdata(PhysStop)
        vl2.set_xdata(-PhysStop)
    fig.canvas.draw_idle()
    
sE.on_changed(update)
sH.on_changed(update)
#samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Find E', color=axcolor, hovercolor='0.975')

resetax2 = plt.axes([0.6, 0.025, 0.2, 0.04])
button2 = Button(resetax2, 'Go to next E', color=axcolor, hovercolor='0.975')

Es = []

def find_next(event):
    global E, Es
    initE = E
    Es = findEs()
    print(Es)
    best = 0
    minDist = +math.inf
    for pE in Es:
        if abs(initE - pE) < minDist:
            minDist = abs(initE - pE)
            best = pE
    E = best
    sE.set_val(E)
    update(0)

def go_to_next(event):
    global E, Es
    initE = E
    best = 0
    minDist = +math.inf
    for pE in Es:
        if abs(initE - pE) < minDist:
            minDist = abs(initE - pE)
            best = pE
    E = best
    sE.set_val(E)
    update(0)
    
button.on_clicked(find_next)
button2.on_clicked(go_to_next)

rax = plt.axes([0.025, 0.25, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('even', 'odd'), active=0)

rax2 = plt.axes([0.025, 0.75, 0.15, 0.15], facecolor=axcolor)
radio2 = RadioButtons(rax2, ('Well', 'Harm. Osc.', 'Abs.'), active=0)

def parity(label):
    global even
    if label == 'even':
        even = True
    else:
        even = False
    update(0)

def potent(label):
    global Pot
    if label == 'Well':
        Pot = label
    elif label == 'Harm. Osc.':
        Pot = 'Osc'
    else:
        Pot = 'Abs'
    update(0)
    
radio.on_clicked(parity)
radio2.on_clicked(potent)


def dichotomie(Emin, Emax, PosToNeg = True, citer = 0):
    global E
    if citer >= 10000:
        return
    E = (Emax + Emin)/2
    _, f = EulerAndPlot()
    if abs(f[-1]) <= 0.1:
        return
    if PosToNeg:
        if f[-1] < 0:
            return dichotomie(Emin, E, PosToNeg, citer+1)
        else:
            return dichotomie(E, Emax, PosToNeg, citer+1)
    else:
        if f[-1] < 0:
            return dichotomie(E, Emax, PosToNeg, citer+1)
        else:
            return dichotomie(Emin, E, PosToNeg, citer+1)

#resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(resetax, 'Find E', color=axcolor, hovercolor='0.975')

def findEs():
    global E
    prevS = 0
    prevE = 0
    Es = []
    xtab, f = EulerAndPlot()
    for BigE in np.linspace(0, 30, 1000): #Broad Phase
        E = BigE
        xtab, f = EulerAndPlot()
        if np.sign(f[-1]) != prevS:
            if prevS == 0:
                prevS = np.sign(f[-1])
            else:
                #print('Specific')
                dichotomie(prevE, E, prevS == ((prevS+1)/2 == 1), 0)
                Es.append(E)
        #idx = np.argwhere(np.diff(np.sign(np.array([E for _ in xtab]) - np.array([V(x) for x in xtab])))).flatten()
        #if idx.size == 0:
        #    print('Stopped at: ', E)
        #    break
        prevE = E
        prevS = np.sign(f[-1])
    return Es

#button.on_clicked(find_next)

plt.show()

