## 指令

```c++
#pragma omp 指令 [字句[[,] 字句]...]
{
  ...  
}
```

* `parallel` 开始并行执行语句
* `for` 在多个线程中并行执行 for 循环
* `sections` 包含多个可以并行执行的 sections 结构
* `single` 单线程执行（不一定是主线程）
* `master` 主线程执行
* `cirital` 任意时刻只能被单个线程执行
* `barrier` 指定屏障，用于同步所有线程
* `taskwait` 等待子线程完成
* `atomic` 确保指定内存位置会原子更新
* `flush` 使线程当前内存数据与实际内存数据一致
* `ordered` 并行执行的 for 循环将按照循环体变量循序执行
* `threadprivate` 指定变量为本地存储

