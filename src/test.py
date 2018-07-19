#coding:UTF-8

class Employee:
  '所有员工的基类'
  empCount = 0  # 这是一个类变量

  def __init__(self, name, salary):
    self.name = name
    self.salary = salary
    Employee.empCount += 1

def main():
  emp = Employee('yin',11)
  print(emp.empCount)
if __name__=='__main__':
  l=list(20)
  l.append(14)
  l[10]=9
  print(l)
  main()