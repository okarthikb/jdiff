## jdiff

A very minimal autodiff implementation in Julia. A simple example:

```julia
a = Node(2.0)
b = Node(3.0)
c = Node(4.0)
d = c * c * a^b

backward(d)

print("a.grad = ", a.grad, "\n")
print("b.grad = ", b.grad, "\n")
print("c.grad = ", c.grad, "\n")
```

This outputs:

```
a.grad = 192.0
b.grad = 88.722839111673
c.grad = 64.0
```
