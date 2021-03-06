__Author__ = "xq"

import cv2

cap = cv2.VideoCapture(0)
print(cap.get(3))
print(cap.get(4))
print(cap.get(5))

# cap.set(cv2.CAP_PROP_FPS, 15)

while True:
    _, frame = cap.read()
    # print(frame.dtype)
    cv2.imshow('frame', frame)

    '''waitKey(int delay)这个函数接收一个整型值，如果这个值是零，那么函数不会有返回值，
    如果delay大于0，那么超过delayms后，如果没有按键，那么会返回-1，如果按键那么会返回键盘值。 
    在某些系统中，返回的键盘值可能不是ASCII编码的，所以通过与运算只取字符最后一个字节'''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

