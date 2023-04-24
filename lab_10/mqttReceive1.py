import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    print("Message Recieved: "+message.payload.decode())

def on_connect(client, userdata, flags, rc):
    print("Connected With Result Code "+str(rc))

client = mqtt.Client(transport="websockets")
client.on_message = on_message
client.on_connect = on_connect
client.connect("test.mosquitto.org",8080)
client.subscribe("COMP3004", qos=1)

client.loop_forever()


