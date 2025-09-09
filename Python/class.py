class Dog():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sit(self):
        print(self.name.title() + ' is now sitting')
    
my_dog = Dog('abc', 12)
my_dog.sit()