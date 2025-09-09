guests = ['a', 'b', 'c', 'd']
print(guests)

absence = guests.pop(2)
print(absence)
print(guests)

guests.insert(0, 'f')
print(guests)

guests.append('g')
print(guests)

# while len(guests) > 2:
#     guests.pop()
# print(guests)

for guest in guests:
    print(guest)

numbers = list(range(1, 6))
print(numbers)
for n in numbers:
    print(n)

cubes = [value**3 for value in range(1, 11)]
print(cubes)

players = ['charles', 'martina', 'michael', 'florence', 'eli'] 
print(players[1:3])
print(players[-3:])

print('-'*40)

alien_0 = {'color': 'green', 'points': 5} 
print(alien_0) 
alien_0['x_position'] = 0 
alien_0['y_position'] = 25 
print(alien_0) 
del alien_0['color']
print(alien_0)
for key, value in alien_0.items(): 
    print(key) 
    print(value) 