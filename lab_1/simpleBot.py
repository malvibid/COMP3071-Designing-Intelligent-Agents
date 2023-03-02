import tkinter as tk
import random
import math
import numpy as np

class Bot():

    def __init__(self,namep):
        self.x = random.randint(100,900)
        self.y = random.randint(100,900)
        self.theta = random.uniform(0.0,2.0*math.pi)
        #self.theta = 0
        self.name = namep
        self.ll = 60 #axle width
        self.vl = 0.0
        self.vr = 0.0

    def draw(self,canvas):
        points = [ (self.x + 30*math.sin(self.theta)) - 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - 30*math.cos(self.theta)) - 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - 30*math.sin(self.theta)) - 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + 30*math.cos(self.theta)) - 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - 30*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + 30*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x + 30*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - 30*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta)  \
                ]
        canvas.create_polygon(points, fill="blue", tags=self.name)

        self.sensorPositions = [ (self.x + 20*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y - 20*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta), \
                                 (self.x - 20*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y + 20*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta)  \
                            ]

        centre1PosX = self.x
        centre1PosY = self.y
        canvas.create_oval(centre1PosX-8,centre1PosY-8,\
                           centre1PosX+8,centre1PosY+8,\
                           fill="gold",tags=self.name)

        wheel1PosX = self.x - 30*math.sin(self.theta)
        wheel1PosY = self.y + 30*math.cos(self.theta)
        canvas.create_oval(wheel1PosX-3,wheel1PosY-3,\
                                         wheel1PosX+3,wheel1PosY+3,\
                                         fill="red",tags=self.name)

        wheel2PosX = self.x + 30*math.sin(self.theta)
        wheel2PosY = self.y - 30*math.cos(self.theta)
        canvas.create_oval(wheel2PosX-3,wheel2PosY-3,\
                                         wheel2PosX+3,wheel2PosY+3,\
                                         fill="green",tags=self.name)

        sensor1PosX = self.sensorPositions[0]
        sensor1PosY = self.sensorPositions[1]
        sensor2PosX = self.sensorPositions[2]
        sensor2PosY = self.sensorPositions[3]
        canvas.create_oval(sensor1PosX-3,sensor1PosY-3, \
                           sensor1PosX+3,sensor1PosY+3, \
                           fill="yellow",tags=self.name)
        canvas.create_oval(sensor2PosX-3,sensor2PosY-3, \
                           sensor2PosX+3,sensor2PosY+3, \
                           fill="yellow",tags=self.name)

    # cf. Dudek and Jenkin, Computational Principles of Mobile Robotics
    def move(self,canvas,dt):
        if self.vl==self.vr:
            R = 0
        else:
            R = (self.ll/2.0)*((self.vr+self.vl)/(self.vl-self.vr))
        omega = (self.vl-self.vr)/self.ll
        ICCx = self.x-R*math.sin(self.theta) #instantaneous centre of curvature
        ICCy = self.y+R*math.cos(self.theta)
        m = np.matrix( [ [math.cos(omega*dt), -math.sin(omega*dt), 0], \
                        [math.sin(omega*dt), math.cos(omega*dt), 0],  \
                        [0,0,1] ] )
        v1 = np.matrix([[self.x-ICCx],[self.y-ICCy],[self.theta]])
        v2 = np.matrix([[ICCx],[ICCy],[omega*dt]])
        newv = np.add(np.dot(m,v1),v2)
        newX = newv.item(0)
        newY = newv.item(1)
        newTheta = newv.item(2)
        newTheta = newTheta%(2.0*math.pi) #make sure angle doesn't go outside [0.0,2*pi)
        self.x = newX
        self.y = newY
        self.theta = newTheta
        if self.vl==self.vr: # straight line movement
            self.x += self.vr*math.cos(self.theta) #vr wlog
            self.y += self.vr*math.sin(self.theta)
        canvas.delete(self.name)
        self.draw(canvas)

    def senseLight(self, registryPassives):
        lightL = 0.0
        lightR = 0.0
        for pp in registryPassives:
            if isinstance(pp,Lamp):
                lx,ly = pp.getLocation()
                distanceL = math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                                       (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )
                distanceR = math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                                       (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )
                lightL += 200000/(distanceL*distanceL)
                lightR += 200000/(distanceR*distanceR)
        return lightL, lightR

    def transferFunction(self, lightL, lightR):
        # wheels not moving - no movement
        self.vl = 0
        self.vr = 0


class Lamp():
    def __init__(self,namep):
        self.centreX = random.randint(100,900)
        self.centreY = random.randint(100,900)
        self.name = namep

    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-10,self.centreY-10, \
                                  self.centreX+10,self.centreY+10, \
                                  fill="yellow",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


def initialise(window):
    window.resizable(False,False)
    canvas = tk.Canvas(window,width=1000,height=1000)
    canvas.pack()
    return canvas

def register(canvas):
    registryActives = []
    registryPassives = []
    noOfBots = 2
    noOfLights = 2
    for i in range(0,noOfBots):
        bot = Bot("Bot"+str(i))
        registryActives.append(bot)
        bot.draw(canvas)
    for i in range(0,noOfLights):
        lamp = Lamp("Lamp"+str(i))
        registryPassives.append(lamp)
        lamp.draw(canvas)
    return registryActives, registryPassives

def moveIt(canvas,registryActives,registryPassives):
    for rr in registryActives:
        lightIntensityL, lightIntensityR = rr.senseLight(registryPassives)
        rr.transferFunction(lightIntensityL, lightIntensityR)
        rr.move(canvas,1.0)
    canvas.after(50,moveIt,canvas,registryActives,registryPassives)

def main():
    window = tk.Tk()
    canvas = initialise(window)
    registryActives, registryPassives = register(canvas)
    moveIt(canvas,registryActives,registryPassives)
    window.mainloop()

main()
