### 1. `matrices_vectors.py`：用迴圈做出矩陣的基本操作。
* **#TODO 任務**：實作純量乘法 、矩陣加法、矩陣與向量的乘積。
* 前兩個函數寫了雙層的 `for` 迴圈，遍歷矩陣的每個元素 (entry) 進行相乘或相加。
* 對於 `matrix_vector_product`，用線性組合的概念，並利用前兩個寫好的函數。所以我寫了一個迴圈，把矩陣 $M$ 的每一行乘上向量 $\vec{v}$ 對應的數值，然後加起來。

### 2. `gauss.py`：實作**高斯消去法** 。
* **#TODO 任務**：實作三個基本列運算：交換、伸縮、平移。
* **列運算**：因為這些函數接收 2D 矩陣，而單一列取出來是 1D 向量，所以我先用 `.reshape(1, cols)` 把它變成2D，丟進自訂函數運算完後，再用 `.flatten()` 壓扁放回原矩陣。
* **消去法**：照線性代數步驟，先找 Pivot ，用列交換把最大值換到對角線；接著用列平移把 Pivot 下方的數字全變成 0；最後再把對角線往上的數字變成 0，並將 Pivot 縮放為 1，最終得到 **RREF**。

### 3. `linear_solver.py`：把題目 $A\vec{x} = \vec{b}$ 合併成增廣矩陣，丟給 `gauss.py` 計算結果。
* **#TODO 任務**：檢查 Consistency，計算答案。
* `test_consistency`：尋找 RREF 矩陣中是否出現 `[0, 0, ..., 0 | 非零數字]` 的矛盾列。如果有，就代表方程式無解，回傳 `False`。
* `generate_solution`：如果是無限多組解，規定要把 Free variables 預設為 `1`。我寫了邏輯去找出 Pivot 所在的欄位（基本變數），並將等號右邊的值減去自由變數的影響，計算出最終的解 $\vec{x}$。
* `solve_linear_equations`：合併 $A$ 和 $b$ 成增廣矩陣，呼叫高斯消去法，檢查一致性，最後產出解答。

### 4. `test_linear_solver.py`
* **#TODO 任務**：計算誤差向量 $A\vec{x} - \vec{b}$ 的 L2 範數 (Euclidean distance)，看它是不是趨近於零。
* 用前面自己建的函式來運算。我先用 `matrix_vector_product(A, x)`，再用 `scalar_matrix(-1.0, b)` 把 $\vec{b}$ 變成負的，最後用 `matrix_sum` 把兩者相加，得出誤差向量。
* 最後用 `torch.norm()` 算出這個誤差向量的長度。

### 疑問：
1. 我該如何確定超大矩陣的答案是否有誤？亂數問題跟浮點數捨入誤差好像都會影響？老師可以提供一些想法嗎，因為前面幾個測試應該可以驗證高斯消去法等應該是對的，但l2 norm卻很大。
3. `gauss.py` 的註解 # Set zero thresh (if entry is larger than this value, treat it as zero. Otherwise, treat it as nonzero) 是寫錯了？
