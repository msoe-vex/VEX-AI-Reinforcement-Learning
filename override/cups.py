class cup:

    def __init__(self, pin1=None, pin2=None, top = True):
        self.color1 = "white"
        self.color2 = "black"

        if top:
            self.top = self.color1
            self.bottom = self.color2
        else:
            self.top = self.color2
            self.bottom = self.color1

        self.pinTop =  pin2
        self.pinBottom = pin1

    def set_pinTop(self, pin):
        self.pinTop = pin

    def set_pinBottom(self, pin):
        self.pinBottom = pin

    def get_pinTop(self):
        return self.pinTop

    def get_pinBottom(self):
        return self.pinBottom



    