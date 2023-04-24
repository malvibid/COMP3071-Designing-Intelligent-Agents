import paho.mqtt.client as mqtt
from pprint import pprint
import random
import time
import math
import sys

def isConvertibleToFloat(value):
  try:
    float(value)
    return True
  except:
    return False

def onMessage(client,userdata,message): #callback
    messageString = message.payload.decode()
    message = list(messageString.split(","))
    #things that look like numbers, become numbers
    for i,s in enumerate(message):
        if isConvertibleToFloat(s):
            message[i] = float(s)
    userdata["messageList"].append(message)

def distance(r,m):
    return math.sqrt((r[1]-m[1])**2 + (r[2]-m[2])**2)

def main():
    currentBotList = []
    currentBotList.append(["Robot1",random.randrange(0,1000),\
                           random.randrange(0,1000), "waiting"])
    currentBotList.append(["Robot2",random.randrange(0,1000),\
                           random.randrange(0,1000), "waiting"])
    currentBotList.append(["Robot3",random.randrange(0,1000),\
                           random.randrange(0,1000), "waiting"])
    currentBotList.append(["Robot4",random.randrange(0,1000),\
                           random.randrange(0,1000), "waiting"])

    #set up mqtt
    studentNumber = "ID00000023"
    myLocation = studentNumber
    mqttMessageList = []
    client = mqtt.Client(transport="websockets",\
                         userdata={"messageList":mqttMessageList})
    client.connect("test.mosquitto.org",8080)
    topic = "COMP3004_"+studentNumber
    client.subscribe(topic, qos=1)
    client.on_message = onMessage
    client.loop_start()

    countdown = random.randrange(4,10)
    #main time loop
    while True:
        print("########################")
        pprint(currentBotList, width=140)
        pprint(mqttMessageList, width=140)

        #bots choose whether to take bid
        for i,r in enumerate(currentBotList):
            if r[3]=="bidding":
                timeOfCleaningRequest = [m[3] for m in mqttMessageList if m[0]=="biddingForCleanup"][0]
                currentBotDistance = [m[3] for m in mqttMessageList if m[0]=="biddingForCleanup" and m[4]==r[0]][0]
                otherBotDistances =[m[3] for m in  mqttMessageList if m[0]=="biddingForCleanup" and m[4]!=r[0]]
                if all(currentBotDistance<d for d in otherBotDistances):
                    message = "claimingBid"+","+str(m[3])+","+str(m[4])+","+str(timeOfCleaningRequest)+","+r[0]
                    client.publish(topic=topic,payload=message,qos=1,retain=False)
                    currentBotList[i][3] = "cleaning"
                    currentBotList[i][1] = m[1]
                    currentBotList[i][2] = m[2]
                    currentBotList[i].append(random.randrange(10,20)) #time to finish cleaning
                else:
                    message = "withdrawingBid"+","+str(m[3])+","+str(m[4])+","+str(timeOfCleaningRequest)+","+r[0]
                    client.publish(topic=topic,payload=message,qos=1,retain=False)
                    currentBotList[i][3] = "waiting"
        time.sleep(0.1) #to allow time for message list to update
        
        #bots monitor traffic
        for i,r in enumerate(currentBotList):
            if r[3]=="waiting":
                for m in mqttMessageList:
                    if m[0]=="cleanupNeeded" and not any( [n[0]=="claimingBid" and n[1]==m[3] for n in mqttMessageList]):
                        message = "biddingForCleanup,"+str(m[3])+","+str(m[4])+","+str(distance(r,m))+","+r[0]
                        client.publish(topic=topic,payload=message,qos=1,retain=False)
                        currentBotList[i][3] = "bidding"        

        #bots doing cleaning
        for i,r in enumerate(currentBotList):
            if r[3]=="cleaning":
                currentBotList[i][4] -= 1
                if currentBotList[i][4]==0:
                    del currentBotList[i][4] #remove cleaning countdown time
                    currentBotList[i][3] = "waiting"

        #does a spillage occur
        if countdown==0:
            x = random.randrange(0,1000)
            y = random.randrange(0,1000)
            print("Spillage at: ",x," ",y)
            message = "cleanupNeeded,"+str(x)+","+str(y)+","+ \
                               str(time.time())+","+str(myLocation)
            client.publish(topic=topic,payload=message,qos=1,retain=False)
        countdown -= 1

        time.sleep(1.0)
                        
main()
