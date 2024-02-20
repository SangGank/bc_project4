from typing import NoReturn
def abc() -> NoReturn:
    print('abc 함수')
    NoReturn
def cde():
    print('cde 함수')
    return NoReturn
    
def efg():
    print('efg 함수')

def main():
    print('이것는 확실히 뜨지?')
    x=abc()
    print(x)
    y= cde()
    print("y의 결과는? ",y)
    # print(abc())
    z= efg()
    print(z)
    print('이거 떠?')
main()