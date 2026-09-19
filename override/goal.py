class goal:

    def __init__(self, toggle, color=2,pin=None):
        self.toggle = toggle
        if color == 1:
            self.color = "red"
        elif color == 0:
            self.color = "blue"
        else:
            self.color = "black"

        self.pinTop =  pin
    
    
    def set_pinTop(self, pin):
        self.pinTop = pin
    
    def get_score(self, color):
        i = self.pinTop
        result = 0
        while i is not None:
            if i.get_color() == color or (i.get_color() == "yellow" and self.toggle.color == color):
                result += i.get_value()
            i = i.nextPin()
        return result

    def getTop(self):
        i = self.pinTop
        while i is not None:
            if i.nextPin() is not None:
                i = i.nextPin() 
            else:
                break
        return i
    
    
    
        