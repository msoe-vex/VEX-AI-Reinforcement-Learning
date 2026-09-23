from override.abstractPin import abstractPin


class red_yellowPin(abstractPin):
    def __init__(self, cup1=None, cup2=None, top=True):
        self.color1 = "red"
        self.color2 = "blue"
        self.value1 = 5
        self.value2 = 5
        if top:
            self.top = self.color1
            self.bottom = self.color2
        else:
            self.top = self.color2
            self.bottom = self.color1
        self.cupTop = cup2
        self.cupBottom = cup1

    def display_info(self):
        return f"Color: {self.color1}, Value: {self.value1}"

    def get_valueTop(self):
        if self.top == self.color1:
            return self.value1
        else:
            return self.value2

    def get_valueBottom(self):
        if self.bottom == self.color1:
            return self.value1
        else:
            return self.value2

    def get_cupTop(self):
        return self.cupTop

    def get_cupBottom(self):
        return self.cupBottom

    def set_cupTop(self, cup):
        self.cupTop = cup

    def set_cupBottom(self, cup):
        self.cupBottom = cup