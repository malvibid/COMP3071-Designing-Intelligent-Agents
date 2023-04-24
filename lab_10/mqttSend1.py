import paho.mqtt.client as mqtt

client = mqtt.Client(transport="websockets")
client.connect("test.mosquitto.org",8080)
client.publish(topic="COMP3004", payload="this is a test message", qos=1, retain=False)

