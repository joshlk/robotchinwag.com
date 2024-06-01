# Partitioning a matmul: considering cache aware sess and parallelisation 

This article summarises what needs to be considered when designing a matrix multiplication algorithm for a system with a memory hierarchy and multiple parallel cores. The aim of the article is to provide an intuition to the reader and purposefully omits many details a matmul planner would have to include.

## Matmul FLOPs

FLOPs is the measurement of the number of floating point operations counted within a period. It's also common to report FLOP/s or floating point operations per second.

Consider a matrix multiplication between matrices $A$ and $B$ with sizes $m$ by $k$ and $k$ by $n$, respectively. For each row in $A$, you apply the dot product with each column in $B$. A dot product of two vectors of size $k$ requires $k$ multiplication operations and $k-1$ addition operations. As a matrix multiplication has $m n$ dot products, the total number of floating point operations are:

$$
\text { FLOPs of } \begin{aligned}
\mathrm{AB} & =k m n+(k-1) m n \\
& \approx 2 k m n
\end{aligned}
$$

## Idealised system

The aim is to model the execution of the GEMM operation $C \leftarrow A B+C$, given the matrices $A$ with the shape $m_{H}$ by $k_{H}$, $B$ with the shape $k_{H}$ by $n_{H}$ and $C$ with the shape $m_{H}$ by $n_{H}$ (the meaning of $\mathrm{H}$ will become clear).

Consider an idealised system with the following features and assumptions:

- The system has a memory hierarchy of $H$ levels, with level $h$ of the hierarchy denoted as $L_{h}$.
- Each level can have multiple workers, each with their own private memory. The number of workers at level $h$ is $w_{h}$.
- The memory capacity of an individual worker at level $h$ is denoted as $S_{h}$. Generally $S_{h}<S_{h+1}<\ldots<S_{H}$.
- Loading a floating point number from level $h+1$ to level $h$ costs time $\rho_{h}$. It's assumed that $\rho_{h+1}<\rho_{h}<\ldots<\rho_{H}$.
- Storing a floating point number from level $h$ to level $h+1$ costs time $\sigma_{h}$. It's assumed that $\sigma_{h}<\sigma_{h+1}<\ldots<\sigma_{H-1}$.
- To accumulate a floating point number using an atomic operation (load and store) from level $h$ to level $h+1$ costs time $\phi_{h}$. It's assumed that $\phi_{h}<\phi_{h+1}<\ldots<\phi_{H-1}$.
- To write a zero value to level $h$ costs time $\alpha_{h}$
- At each level, an optimised GEMM routine is provided: given the matrices $A$ with the shape $m_{h}$ by $k_{h}, B$ with the shape $k_{h}$ by $n_{h}$ and $C$ with the shape $m_{h}$ by $n_{h}$, the operation $C \leftarrow A B+C$ costs time $2 m_{h} k_{h} n_{h} \gamma_{h}$, where $\gamma_{h}$ is the time cost per floating point operation when computed at level $h$. The matrices $A, B$ and $C$ must already be loaded into the memory at level $h$.

![](https://cdn.mathpix.com/cropped/2024_05_24_3ff8bc6a4d86e938a7f8g-1.jpg?height=404&width=779&top_left_y=2075&top_left_x=185)

## Partitioning schemes

At each level of the memory hierarchy, we are going to consider partitioning the matrixes into sub-blocks:

$$
C=\left(\begin{array}{c|c|c}
C_{11} & \cdots & C_{1 N} \\
\hline \vdots & & \vdots \\
\hline C_{M 1} & \cdots & C_{M N}
\end{array}\right), A=\left(\begin{array}{c|c|c}
A_{11} & \cdots & A_{1 K} \\
\hline \vdots & & \vdots \\
\hline A_{M 1} & \cdots & A_{M K}
\end{array}\right), B=\left(\begin{array}{c|c|c}
B_{11} & \cdots & B_{1 N} \\
\hline \vdots & & \vdots \\
\hline B_{K 1} & \cdots & B_{K N}
\end{array}\right)
$$

Above the level $h$ has been omitted for brevity but $M_{h}, K_{h}$ and $N_{h}(\mathrm{M}, \mathrm{K}$ and $\mathrm{N}$ above) are the number of partitions to each axis at level $h$ compared to level $h+1$; while $m_{h}, k_{h}$ and $n_{H}$ are the dimension sizes of the matrix blocks at level $h$. Therefore $M_{h}=m_{h+1} / m_{h}$,

$K_{h}=k_{h+1} / k_{h}$ and $N_{h}=n_{h+1} / n_{h}$.

At a given level $h$, the matrix multiplication $C \leftarrow A B+C$ can then be calculated by using the following:

$$
C_{i j} \leftarrow \sum_{p=1}^{K_{h}} A_{i p} B_{p j}+C_{i j}
$$

Therefore to calculate the complete output matrix $C$, we need to iterate over $M_{h}, K_{h}$ and $N_{h}$ (i, $p$ and $\left.j\right)$. These loops can be nested in any order; however, in practice there are only 3 options considered below. The aim of the game will be to optimise the 3 options with the partitioning parameters $m_{h}, k_{h}$ and $n_{H}$ at each level. This article ignores the possibility of jointly optimising across all levels.

## Algorithm 1: $\mathrm{K}_{\mathrm{h}}$ inner loop

| Algorithm 1 | Cost per execution | Number of executions |
| :---: | :---: | :---: |
| for $j=1, \ldots, N_{h}$ <br> for $i=1, \ldots, M_{h}$ <br> $\qquad$ Load $C_{i j}$ from $L_{h+1}$ to $L_{h}$ <br> for $p=1, \ldots, K_{h}$ <br> Load $A_{i p}$ from $L_{h+1}$ to $L_{h}$ <br> Load $B_{p j}$ from $L_{h+1}$ to $L_{h}$ <br> Update $C_{i j}<-A_{i p} B_{p j}+C_{i j}$ <br> endfor <br> Store $C_{i j}$ from $L_{h}$ to $L_{h+1}$ <br> endfor <br> endfor | $m_{h} n_{h} \rho_{h}$ <br> $m_{h} k_{h} \rho_{h}$ <br> $k_{h} n_{h} \rho_{h}$ <br> $2 m_{h} n_{h} k_{h} \gamma_{h}$ <br> $m_{h} n_{h} \sigma_{h}$ | $M_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} N_{h}$ |

Below we iterate over $K_{h}$ in the inner loop. The ordering of the two outer loops doesn't matter.

In algorithm 1, $C_{i j}$ is independent of the inner loop variable $p$ and so can be loaded and stored once per $N_{h}$ and $M_{h}$ iteration, so the cost to load and store is amortized across the $K_{h}$ loop.

The time cost for updating matrix $C$ at level $h+1$ if algorithm 1 is executed sequentially, as above:

$$
T_{h+1}^{\prime}=M_{h} N_{h} m_{h} n_{h}\left(\rho_{h}+\sigma_{h}\right)+M_{h} K_{h} N_{h}\left(m_{h} k_{h} \rho_{h}+k_{h} n_{h} \rho_{h}+2 m_{h} n_{h} k_{h} \gamma_{h}\right)
$$

As $M_{h}=m_{h+1} / m_{h}, K_{h}=k_{h+1} / k_{h}$ and $N_{h}=n_{h+1} / n_{h}$ you can simplify the above to:

$$
T_{h+1}^{\prime}=m_{h+1} n_{h+1}\left(\rho_{h}+\sigma_{h}\right)+m_{h+1} k_{h+1} n_{h+1}\left(\frac{\rho_{h}}{n_{h}}+\frac{\rho_{h}}{m_{h}}+2 \gamma_{h}\right)
$$

Instead of executing the inner-most loop sequentially, we can execute the body in parallel over multiple workers. If $K_{h}>>w_{h}$ then each worker can execute approximately $K_{h} / w_{h}$ of the iterations. However, there is an additional cost as each worker will need to replace the
update $C_{i j}$ using an atomic operation to prevent data corruption and load $C_{i j}$ with creating matrix of zeros of the same size. This means we replace $\rho_{h}+\sigma_{h}$ with $\alpha_{h}+\phi_{h}$, giving us:

$$
T_{h+1}=m_{h+1} n_{h+1}\left(\alpha_{h}+\phi_{h}\right)+\frac{m_{h+1} k_{h+1} n_{h+1}}{w_{h}}\left(\frac{\rho_{h}}{n_{h}}+\frac{\rho_{h}}{m_{h}}+2 \gamma_{h}\right)
$$

To determine time cost per operation, we use the fact that $T_{h+1}=2 m_{h+1} k_{h+1} n_{h+1} \gamma_{h+1}$ :

$$
\gamma_{h+1}=\frac{\alpha_{h}+\phi_{h}}{2 k_{h+1}}+\frac{1}{w_{h}}\left(\frac{\rho_{h}}{2 n_{h}}+\frac{\rho_{h}}{2 m_{h}}+\gamma_{h}\right)
$$

In the above algorithm, we choose how we partition the 3 matrices by tunning $m_{h}, n_{h}$ and $k_{h}$ and we want to pick values that minimise the cost $\gamma_{h+1}$, however, we are constrained by the size of the memory at that level so that $m_{h} n_{h}+m_{h} k_{h}+k_{h} n_{h} \leq S_{h}$. Notice how the cost (above equation) is independent of $k_{h}$; therefore, we can make $k_{h}$ as small as needed. This allows us to pick values of $m_{h}$ and $n_{h}$, which are as large as possible to fit into the memory. The minimum can be shown to be obtained when $m_{h} \approx n_{h} \approx \sqrt{S_{h}}$ (see appendix).

This strategy implies prioritising making the blocks of $C\left(C_{i j}\right)$ as large as possible and keeping them square (by increasing $m_{h}$ and $n_{h}$, and making $k_{h}$ small). If we can fit $C$ into memory without partitioning it, then we make the blocks of $A$ and $B\left(A_{i p}\right.$ and $\left.B_{p j}\right)$ as large as possible by increasing $k_{h}$.

## Algorithm 2: $\mathbf{N}_{\mathrm{h}}$ inner loop

| Algorithm 2 | Cost per execution | Number of executions |
| :---: | :---: | :---: |
| for $p=1, \ldots, K_{h}$ <br> for $i=1, \ldots, M_{h}$ <br> Load $A_{i p}$ from $L_{h+1}$ to $L_{h}$ <br> for $j=1, \ldots, N_{h}$ <br> Load $C_{i j}$ from $L_{h+1}$ to $L_{h}$ <br> Load $B_{p j}$ from $L_{h+1}$ to $L_{h}$ <br> Update $C_{i j}<-A_{i p} B_{p j}+C_{i j}$ <br> Store $C_{i j}$ from $L_{h}$ to $L_{h+1}$ <br> endfor <br> endfor <br> endfor | $m_{h} k_{h} \rho_{h}$ <br> $m_{h} n_{h} \rho_{h}$ <br> $k_{h} n_{h} \rho_{h}$ <br> $2 m_{h} n_{h} k_{h} v_{h}$ <br> $m_{h} n_{h} \sigma_{h}$ | $M_{h} K_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ |

We are now investigating having $N_{h}$ as the innermost loop. Again, the ordering of the two outermost loops is inconsequential.

Doing a similar analysis as above, you can determine the time cost per operation as:

$$
\gamma_{h+1}=\frac{\rho_{h}}{2 n_{h+1}}+\frac{1}{w_{h}}\left(\frac{\alpha_{h}+\phi_{h}}{2 k_{h}}+\frac{\rho_{h}}{2 m_{h}}+\gamma_{h}\right)
$$

Notice that the equation it is independent of $n_{h}$. Again, we want to find $m_{h}, n_{h}$ and $k_{h}$ which minimises the cost with the constraint that $m_{h} n_{h}+m_{h} k_{h}+k_{h} n_{h} \leq S_{h}$. It can be shown that the minimum is obtained when $m_{h} \approx k_{h} \approx \sqrt{S_{h}}$.

The implication is that you prioritise making the blocks of $A\left(A_{i p}\right)$ as large as possible and keeping them square (by increasing $m_{h}$ and $k_{h}$, and making $n_{h}$ small). If we can fit $A$ into memory without partitioning it, then we make the blocks of $A$ and $B\left(A_{i p}\right.$ and $\left.B_{p j}\right)$ as large as possible by increasing $n_{h}$.

## Algorithm 3: $\mathrm{M}_{\mathrm{h}}$ inner loop ordering

We are now investigating having $M_{h}$ as the innermost loop. Again, the ordering of the two outermost loops is inconsequential.

| Algorithm 3 | Cost per execution | Number of executions |
| :---: | :---: | :---: |
| for $j=1, \ldots, N_{h}$ <br> {f9ac94333-fd2c-4178-80c2-04dd601e29a4}for $p=1, \ldots, K_{h}$ <br> Load $B_{p j}$ from $L_{h+1}$ to $L_{h}$ <br> for $i=1, \ldots, M_{h}$ <br> Load $C_{i j}$ from $L_{h+1}$ to $L_{h}$ <br> Load $A_{i p}$ from $L_{h+1}$ to $L_{h}$ <br> Update $C_{i j}<-A_{i p} B_{p j}+C_{i j}$ <br> Store $C_{i j}$ from $L_{h}$ to $L_{h+1}$ <br> endfor <br> endfor <br> endfor | $k_{h} n_{h} \rho_{h}$ <br> $m_{h} n_{h} \rho_{h}$ <br> $m_{h} k_{h} \rho_{h}$ <br> $2 m_{h} n_{h} k_{h} \gamma_{\mathrm{h}}$ <br> $m_{h} n_{h} \sigma_{h}$ | $N_{h} K_{h}$ <br>  <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ <br> $M_{h} K_{h} N_{h}$ |

Doing a similar analysis as above, you can determine the time cost per operation as:

$$
\gamma_{h+1}=\frac{\rho_{h}}{2 m_{h+1}}+\frac{1}{w_{h}}\left(\frac{\alpha_{h}+\phi_{h}}{2 k_{h}}+\frac{\rho_{h}}{2 n_{h}}+\gamma_{h}\right)
$$

Notice that it is independent of $m_{h}$. Again, we want to find $m_{h}, n_{h}$ and $k_{h}$ which minimises the cost with the constraint that $m_{h} n_{h}+m_{h} k_{h}+k_{h} n_{h} \leq S_{h}$. It can be shown that the minimum is obtained when $k_{h} \approx n_{h} \approx \sqrt{S_{h}}$.

The implication is that you prioritise making the blocks of $B\left(B_{p j}\right)$ as large as possible and keeping them square (by increasing $k_{h}$ and $n_{h}$, and making $m_{h}$ small). If we can fit $B$ into memory without partitioning it, then we make the blocks of $A$ and $C$ ( $A_{i p}$ and $C_{i j}$ ) as large as possible by increasing $m_{h}$.

## Summary

| Algorithm | Innermost dim | Fit into memory | Part A: maximise | Part B: maximise |
| :--- | :--- | :--- | :--- | :--- |
| 1 | K | C | $m, n ;$ given $k=1$ | $k$ |
| 2 | N | A | $m, k ;$ given $n=1$ | $n$ |
| 3 | M | B | k, $n ;$ given $m=1$ | $m$ |

## Application to Izanagi

For an Izanagi chiplet we can model the memory hierarchy as follows:

| Name | Level (h) | Memory capacity per worker | Workers |
| :--- | :--- | :--- | :--- |
| Main memory <br> (HBM or LPDDR) | 5 | - |  |
| SLC |  |  | 1 |
| LCC | 4 | $128 \mathrm{MiB}^{\star}$ | 8 |
| L2 | 3 | $16 \mathrm{MiB}^{*}$ | 32 |

*Assuming 256 compute tiles, SLC and LCC are split $50: 50$ and 8 NUCA zones with 32 compute tiles each.

For a given matmul at each level of the hierarchy, we can either use algorithms 1, 2 or 3 . As a rule of thumb, we want to pick the algorithm that is going to keep the largest matrix in memory at that level. We also need to keep in mind that we want to keep any given dimension
greater than or equal to the 128, referred to as the "grain size". Blocks of 128 utilise the Neural Engines's convolution engine computation power and we don't want to go smaller than the system's cache line size of 64 bytes.

For example, let's consider a large matrix multiplication whereby $m_{5}=4,096, k_{5}=4,096$ and $n_{5}=16,384$. If we use FP8 (1B per number),

$A=17$ MiB, $B=67$ MiB, $C=67$ MiB. Reasoning per level:

- Level 4: The memory capacity is $123 \mathrm{MiB}$ and the largest matrix to fit is matrix C, so we use algorithm 1. To maximise the memory usage we use $K=2, \mathrm{k}_{\mathrm{h}}=2,048$.
- Level 3: The memory capacity is $16 \mathrm{MiB}$ and the largest matrix to fit is matrix $\mathrm{A}$, so we use algorithm 2. To maximise the memory usage we use $N=16, n_{h}=1,024$.
- Level 2: The memory capacity is $2 \mathrm{MiB}$ and the largest matrix to fit is matrix $\mathrm{B}$, so we use algorithm 3 . To maximise the memory usage we use $M=32 m_{h}=128$.

The strategy is summarised in this table (" $\mathrm{A}_{(\mathrm{h}}$ " is the block size of matrix $\mathrm{A}$ at level $\mathrm{h}$ ):

| Name | Level (h) | Algorithm | $m_{h}$ | $k_{h}$ | $n_{h}$ | $A_{(h)}$ | $\mathbf{B}_{(\mathrm{h})}$ | $C_{(h)}$ | $A_{(h)}+B_{(h)}+C_{(h)}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Main memory | 5 | - | 4,096 | 4,096 | 16,384 | $17 \mathrm{E}+6$ | $67 \mathrm{E}+6$ | $67 \mathrm{E}+6$ | $151 E+6$ |
| SLC | 4 | 1 | 4,096 | 2,048 | 16,384 | $8 \mathrm{E}+6$ | $34 E+6$ | $67 \mathrm{E}+6$ | $109 \mathrm{E}+6$ |
| LCC | 3 | 2 | 4,096 | 2,048 | 1,024 | $8 \mathrm{E}+6$ | $2 E+6$ | $4 E+6$ | $15 \mathrm{E}+6$ |
| L2 | 2 | 3 | 128 | 2,048 | 1,024 | $262 E+3$ | $2 E+6$ | $131 E+3$ | $2 E+6$ |

## Limitations and other considerations

Other considerations that have not been mentioned above:

- In the idealised model, the matmul cost is linear to the dimension sizes of the 3 matrices, while in practice, the costs will vary nonlinearly. For example, larger matrices or matrices of certain shape multiples can be cheaper to compute. There are also costs regarding moving the data between components and startup costs.
- Tensors, e.g. activations, could already be stored in the memory hierarchy from a previous operation, and so you would want to take advantage of this when planning a matmul.
- Memory layout of tensors in main memory to utilise maximum bandwidth.
- Memory layout of tensors supplied to the compute tiles to obtain maximum compute capacity.
- The neural engine uses double buffering to overlap read/writes with compute while we model it sequentially above.
- We don't jointly optimise all levels of the hierarchy together


## Appendix

## Minimising the cost for Algorithm 1

We approximate that $k_{h}$ is very small and we want to maximise $m_{h}$ and $n_{h}$ :

$$
m_{h} n_{h} \approx S_{h}
$$

The cost then is proportional to:

$$
Q=\frac{1}{m_{h}}+\frac{1}{n_{h}}
$$

After substituting the previous equation:

$$
Q=\frac{n_{h}}{S_{h}}+\frac{1}{n_{h}}
$$

The gradient being:

$$
\frac{\mathrm{d} Q}{\mathrm{~d} n_{h}}=\frac{1}{S_{h}}-\frac{1}{n_{h}^{2}}
$$

To find the minimum of $\mathrm{Q}$ we solve for when the gradient is 0 , which gives:

$$
n_{h}=m_{h}=\sqrt{S_{h}}
$$

## Sources

- Heavily inspired by A Family of High-Performance Matrix Multiplication Algorithm (2004)

