# Quick I2C scan across multiple pin pairs
from machine import SoftI2C, Pin

pairs = [(8,9), (41,42), (1,2), (4,5), (6,7), (10,11), (12,13), (14,15)]
for sda, scl in pairs:
    try:
        i2c = SoftI2C(sda=Pin(sda), scl=Pin(scl), freq=100000)
        devs = i2c.scan()
        if devs:
            print("FOUND GPIO%d/GPIO%d: %s" % (sda, scl, [hex(d) for d in devs]))
    except Exception as e:
        print("GPIO%d/GPIO%d: ERROR %s" % (sda, scl, e))
