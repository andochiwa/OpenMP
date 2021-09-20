## 指令

```c++
#pragma omp 指令 [字句[[,] 字句]...]
{
  ...  
}
```

| 指令          | 描述                                        |
| ------------- | ------------------------------------------- |
| parallel      | 开始并行执行语句                            |
| for           | 在多个线程中并行执行 for 循环               |
| sections      | 包含多个可以并行执行的 sections 结构        |
| single        | 单线程执行（不一定是主线程）                |
| master        | 主线程执行                                  |
| cirtial       | 任意时刻只能被单个线程执行                  |
| barrier       | 指定屏障，用于同步所有线程                  |
| taskwait      | 等待子线程完成                              |
| atomic        | 确保指定内存位置会原子更新                  |
| flush         | 使线程当前内存数据与实际内存数据一致        |
| ordered       | 并行执行的 for 循环将按照循环体变量循序执行 |
| threadprivate | 指定变量为本地存储                          |

| 字句         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| defaul       | 控制 parallel 或 task 结构中的变量数据共享属性               |
| shared       | parallel 或 task 结构中一个或多个变量为共享变量              |
| private      | 一个或多个变量为本地变量                                     |
| firstprivate | 一个或多个变量为本地变量并且初始值为并行结构执行前的值       |
| lastprivate  | 一个或多个变量为本地变量并且值为并行结构执行后的值           |
| reduction    | 一个或多个变量为本地变量但是初始值根据不同运算符决定，<br />执行完成后变量值会被更新 |
| copyin       | 使线程本地变量值域主线程变量值相同                           |
| copyprivate  | 使属于 parallel 区域的变量的值在不同线程中相同               |
| schedule     | 设置 for 循环并行执行方式 dynamic、guided、runtime 和 static |
| num_threads  | 线程数目                                                     |
| if           | 并行语句执行条件                                             |
| notwait      | 忽略线程同步等待                                             |

| API                                        | 描述                             |
| ------------------------------------------ | -------------------------------- |
| void omp_set_num_threads(int num_threads); | 设置 parallel 区域线程数目       |
| int omp_get_num_threads();                 | 获取线程数目                     |
| int omp_get_max_threads();                 | 获取最大线程数目                 |
| int omp_get_thread_num();                  | 获取可用处理器数目               |
| int omp_int_parallel();                    | 返回为 true 代表当前处于并行区域 |
| void omp_set_dynamic(int dynamic_threads); | 允许或禁用动态线程调整           |
| int omp_get_dynamic();                     | 获取动态线程调整状态，允许或禁用 |
| void omp_set_nested(int nested);           | 运行或禁用嵌套并行结构           |
| int omp_get_nested(void);                  | 返回并行结构嵌套状态，允许或禁用 |

