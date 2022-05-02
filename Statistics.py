#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
frequency1=0
frequency2=0
frequency3=0
frequency4=0
frequency5=0
frequency6=0
frequency7=0
frequency8=0
frequency9=0
frequency10=0
frequency11=0
frequency12=0
trials=6_000_000
for roll in range(trials):
    face=random.randrange(1,7)+random.randrange(1,7)
    if face==1:
        frequency1 += 1
    elif face==2:
        frequency2 += 1
    elif face==3:
        frequency3 += 1
    elif face==4:
        frequency4 += 1
    elif face==5:
        frequency5 += 1
    elif face==6:
        frequency6 += 1
    elif face==7:
        frequency7 += 1
    elif face==8:
        frequency8 += 1
    elif face==9:
        frequency9 += 1
    elif face==10:
        frequency10 += 1
    elif face==11:
        frequency11 += 1
    elif face==12:
        frequency12 += 1
print(f'Face{"Frequency":>13}')
print(f'{1:>4}{frequency1:>13}')
print(f'{2:>4}{frequency2:>13}')
print(f'{3:>4}{frequency3:>13}')
print(f'{4:>4}{frequency4:>13}')
print(f'{5:>4}{frequency5:>13}')
print(f'{6:>4}{frequency6:>13}')
print(f'{7:>4}{frequency7:>13}')
print(f'{8:>4}{frequency8:>13}')
print(f'{9:>4}{frequency9:>13}')
print(f'{10:>4}{frequency10:>13}')
print(f'{11:>4}{frequency11:>13}')
print(f'{12:>4}{frequency12:>13}')

craps= frequency2 + frequency3 + frequency12
wins= frequency7 + frequency11
print('Craps:',craps/trials)
print('Wins:', wins/trials)


# In[6]:


import random
def roll_dice():
    die1=random.randrange(1,7)
    die2=random.randrange(1,7)
    return(die1, die2)
def display_dice(dice):
    die1, die2 = dice
    print(f'Player rolled {die1}+{die2}={sum(dice)}')
die_values=roll_dice()
display_dice(die_values)
sum_of_dice=sum(die_values)

if sum_of_dice in (7,11):
    game_status='WON'
elif sum_of_dice in (2, 3, 12):
    game_status='LOST'
else:
    game_status='CONTINUE'
    my_points=sum_of_dice
    print('Point is', my_points)
while game_status == 'CONTINUE':
    die_values=roll_dice()
    display_dice(die_values)
    sum_of_dice=sum(die_values)
    
    if sum_of_dice==my_points:
        game_status='WON'
    elif sum_of_dice==7:
        game_status = 'LOST'
        
if game_status=='WON':
    print('Player wins')
else:
    print('Player loses')


# In[7]:


student=('Sue', [89, 94, 85])
student


# In[8]:


name, grade = student


# In[10]:


print(f'{name}:{grade}')


# In[12]:


import math
math.sqrt(900)


# In[13]:


math.floor(9.2)


# In[14]:


get_ipython().run_line_magic('pinfo', 'math.fabs')


# In[15]:


def rectangle_area(length=2, width=3):
    """Return a rectangle's area."""
    return length * width


# In[16]:


rectangle_area()


# In[17]:


rectangle_area(10)


# In[18]:


rectangle_area(10,5)


# In[19]:


def rectangle_area(length, width):
    """Return a rectangle's area"""
    return length * width


# In[20]:


rectangle_area(widgth=5, length=10)


# In[21]:


rectangle_area(width=5, length=10)


# In[22]:


def average(*args):
    return sum(args)/len(args)


# In[23]:


average(5,10)


# In[24]:


average(5,10,15)


# In[25]:


average(5, 10, 15, 20)


# In[26]:


grades=[88, 75, 96, 44, 83]
average(*grades)


# In[27]:


def calculate_product(*args):
    product=1
    for value in args:
        product *= value
    return product


# In[28]:


calculate_product(10, 20, 30)


# In[29]:


calculate_product(1, 6, 22)


# In[30]:


s='Hello'
s.lower()


# In[31]:


s.upper()


# In[32]:


s


# In[33]:


x=7


# In[34]:


def access_global():
    print('x printed from access_global:', x)
access_global()


# In[35]:


def try_to_modify_global():
    x=3.5
    print('x printed from try_to_modify_global:', x)
try_to_modify_global()


# In[36]:


x


# In[38]:


def modify_global():
    global x
    x='hello'
    print('x printed from modify_global:', x)
modify_global()


# In[39]:


x


# In[40]:


from math import ceil, floor


# In[41]:


ceil(10.3)


# In[42]:


floor(10.7)


# In[43]:


import statistics as stats


# In[44]:


grades=[85, 93, 45, 87, 93]
stats.mean(grades)


# In[45]:


import decimal as dec
dec.Decimal('2.5')**2


# In[46]:


x=7


# In[47]:


id(x)


# In[48]:


def cube(number):
    print('id(number)', id(number))
    return number ** 3
cube(x)


# In[49]:


def cube(number):
    print('number is x', number is x)
    return number ** 3
cube(x)


# In[50]:


def cube(number):
    print('id(number) before modifying number:', id(number))
    number **= 3
    print('id(number) after modifying number:', id(number))
    return number
cube(x)


# In[51]:


print(f'x={x}; id(x)={id(x)}')


# In[52]:


width=15.5
print('id:', id(width), 'value:', width)


# In[53]:


width = width * 3
print('id:', id(width), 'value:', width)


# In[77]:


amount=100.0
taxPercent=7.5
tipPercent=20.0
class Purchase(object):
    def _init_(person, amount):
        amount=self.amount
    def calculateTax(self, taxPercent):
        return self.amount*taxPercent/100.0
    def calculateTip(self, tipPercent):
        return self.amount*tipPercent/100.0
    def calculateTotal(self, taxPercent, tipPercent):
        return self.amount * (1 + taxPercent/100.0 + tipPercent/100.0)

tax=Purchase.calculateTax(amount, taxPercent)
tip=Purchase.calculateTip(amount, tipPercent)
total=Purchase.calculateTotal(amount, taxPercent, tipPercent)
print('Tax:', tax)
print('Tip:', tip)
print('Total:', total)
    


# In[86]:


amount=100.0
taxPercent=7.5
tipPercent=20.0
class Purchase(object):
    def _init_(person, amount):
        amount=self_amount
    def calculateTax(amount, taxPercent):
        return amount * taxPercent/100.0
    def calculateTip(amount, tipPercent):
        return amount * tipPercent/100.0
    def calculateTotal(amount, taxPercent, tipPercent):
        return amount * (1+ taxPercent/100.0 + tipPercent/100.0)
tax=Purchase.calculateTax(100.0, 7.5)
tip=Purchase.calculateTip(100.0, 20.0)
total=Purchase.calculateTotal(100.0, 7.5, 20.0)
print('Tax', tax)
print('Tip', tip)
print('Total', total)


# In[87]:


import statistics
statistics.pvariance([1, 3, 4, 2, 6, 5, 3, 4, 5, 2])


# In[89]:


statistics.pstdev([1, 3, 4, 2, 6, 5, 3, 4, 5, 2])


# In[90]:


values=[84, 92, 76, 72, 100, 92, 60, 84, 82, 98, 82, 78, 70, 90, 82, 90, 88, 84, 100, 100, 78, 90, 94, 88, 80, 82, 90, 62, 92, 96, 70, 94, 64, 80, 0, 94, 60, 82, 82, 78, 92, 94, 94, 88, 96, 100, 72, 100, 72, 82, 92, 94, 88, 96, 88, 70, 100, 96, 92, 46, 80, 78, 86, 64, 68, 96, 12, 70, 88, 78, 62, 78, 94, 100, 100, 96, 80, 72, 86, 82, 78, 88, 45, 64, 84, 90, 90, 100, 86, 84, 92, 84, 100, 80, 82, 86, 88, 84, 96, 76, 100, 92, 72, 90, 92, 84, 98, 80, 92]
statistics.pvariance(values)


# In[91]:


statistics.pstdev(values)


# In[ ]:




