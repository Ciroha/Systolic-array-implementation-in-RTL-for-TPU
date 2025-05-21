import numpy as np

def to_signed_binary(number, bits):
    """
    将有符号整数转换为指定位数的二进制补码字符串。

    参数:
    number (int): 要转换的整数。
    bits (int): 二进制表示的位数。

    返回:
    str: 数字的二进制补码表示。
    """
    # 定义目标位数的有符号整数范围
    min_val_target_bits = -2**(bits-1)
    max_val_target_bits = 2**(bits-1) - 1

    # 检查数字是否超出目标位数的表示范围，如果超出则进行截断
    if number > max_val_target_bits:
        # print(f"警告: 数字 {number} 超出 {bits}位有符号正范围，将截断为最大值 {max_val_target_bits}。")
        number_to_format = max_val_target_bits
    elif number < min_val_target_bits:
        # print(f"警告: 数字 {number} 超出 {bits}位有符号负范围，将截断为最小值 {min_val_target_bits}。")
        number_to_format = min_val_target_bits
    else:
        number_to_format = number

    # 转换为二进制补码
    if number_to_format >= 0:
        # 对于正数和零
        return format(number_to_format, f'0{bits}b')
    else:
        # 对于负数，计算二进制补码
        # (1 << bits) + number_to_format 适用于在 'bits' 位内表示负数补码
        return format((1 << bits) + number_to_format, f'0{bits}b')

def generate_and_save_data(
    n,
    file_prefix="matrix",
    generate_matrix_A=True,
    generate_matrix_B=True,
    perform_matrix_multiplication=True,
    perform_vector_multiplication=True
):
    """
    生成矩阵、可选的向量，可选地计算它们的乘积，并将它们以二进制格式保存到文本文件中。
    - 可选生成第一个矩阵 (A) n x 8 (元素转为8位有符号二进制)。
    - 可选生成第二个矩阵 (B) 8 x n (元素转为8位有符号二进制)。
    - 可选计算第三个矩阵 (C) = B @ A (8 x 8) (元素转为16位有符号二进制)。
    - 可选生成一个向量 (V) 8 x 1 (元素转为8位有符号二进制)。
    - 可选计算矩阵-向量乘积 (AV) = A @ V (n x 1) (元素转为16位有符号二进制)。
    二进制数据在文件中无分隔符连接。

    参数:
    n (int): 矩阵维度中的变量 n。
    file_prefix (str): 保存文件的前缀名。
    generate_matrix_A (bool): 是否生成并保存矩阵 A。
    generate_matrix_B (bool): 是否生成并保存矩阵 B。
    perform_matrix_multiplication (bool): 是否执行矩阵B和A的乘法并保存结果C。
    perform_vector_multiplication (bool): 是否生成向量V，执行A和V的乘法并保存V和AV。
    """
    # 检查 n 是否为正整数
    if not isinstance(n, int) or n <= 0:
        print("错误：n 必须是一个正整数。")
        return

    print(f"定义 n = {n}")

    # 定义8位有符号整数的范围
    min_val_8bit = -128
    max_val_8bit = 127

    matrix_A = None
    matrix_B = None

    if generate_matrix_A:
        # 1. 生成第一个随机整数矩阵 A (n x 8)
        matrix_A = np.random.randint(min_val_8bit, max_val_8bit + 1, size=(n, 8), dtype=np.int16)
        print("\n第一个矩阵 A ({} x 8):".format(n))
        print(matrix_A)
        file_A_bin = f"{file_prefix}_A_{n}x8_8bit_binary_no_space.txt"
        try:
            with open(file_A_bin, 'w') as f_handle:
                for row in matrix_A:
                    binary_row = [to_signed_binary(val, 8) for val in row]
                    f_handle.write(''.join(binary_row) + '\n')
            print(f"\n矩阵 A (8位二进制, 无分隔) 已保存到 {file_A_bin}")
        except Exception as e:
            print(f"\n保存矩阵 A 时出错: {e}")


    if generate_matrix_B:
        # 2. 生成第二个随机整数矩阵 B (8 x n)
        matrix_B = np.random.randint(min_val_8bit, max_val_8bit + 1, size=(8, n), dtype=np.int16)
        print("\n第二个矩阵 B (8 x {}):".format(n))
        print(matrix_B)
        file_B_bin = f"{file_prefix}_B_8x{n}_8bit_binary_no_space.txt"
        try:
            with open(file_B_bin, 'w') as f_handle:
                for row in matrix_B:
                    binary_row = [to_signed_binary(val, 8) for val in row]
                    f_handle.write(''.join(binary_row) + '\n')
            print(f"矩阵 B (8位二进制, 无分隔) 已保存到 {file_B_bin}")
        except Exception as e:
            print(f"\n保存矩阵 B 时出错: {e}")


    if perform_matrix_multiplication:
        if matrix_A is None or matrix_B is None:
            print("\n错误：要执行矩阵乘法 (B@A)，必须首先生成矩阵 A 和矩阵 B。将跳过此操作。")
            print("请设置 generate_matrix_A=True 和 generate_matrix_B=True。")
        else:
            # 3. 计算第三个矩阵 C = B @ A (8 x 8)
            matrix_C_product = np.matmul(matrix_B.astype(np.int32), matrix_A.astype(np.int32))
            print("\n第三个矩阵 C (B @ A) (8 x 8) (dtype: {}):".format(matrix_C_product.dtype))
            print(matrix_C_product)
            file_C_bin = f"{file_prefix}_C_product_BxA_8x8_16bit_binary_no_space.txt" #文件名稍作修改以反映BxA
            try:
                with open(file_C_bin, 'w') as f_handle:
                    for row in matrix_C_product:
                        binary_row = [to_signed_binary(val, 16) for val in row]
                        f_handle.write(''.join(binary_row) + '\n')
                print(f"矩阵 C (乘积 BxA, 16位二进制, 无分隔) 已保存到 {file_C_bin}")
            except Exception as e:
                print(f"\n保存矩阵 C 时出错: {e}")

    if perform_vector_multiplication:
        if matrix_A is None:
            print("\n错误：要执行矩阵向量乘法 (A@V)，必须首先生成矩阵 A。将跳过此操作。")
            print("请设置 generate_matrix_A=True。")
        else:
            # 4. 生成随机向量 V (8 x 1)
            vector_V = np.random.randint(min_val_8bit, max_val_8bit + 1, size=(8, 1), dtype=np.int16)
            print("\n随机向量 V (8 x 1):")
            print(vector_V)
            file_V_bin = f"{file_prefix}_vector_V_8x1_8bit_binary_no_space.txt"
            try:
                with open(file_V_bin, 'w') as f_handle:
                    for val_array in vector_V: # vector_V is (8,1)
                        binary_val = to_signed_binary(val_array[0], 8)
                        f_handle.write(binary_val + '\n')
                print(f"向量 V (8位二进制, 无分隔, 每元素一行) 已保存到 {file_V_bin}")
            except Exception as e:
                print(f"\n保存向量 V 时出错: {e}")

            # 5. 计算矩阵-向量乘积 AV = A @ V (n x 1)
            result_vector_AV = np.matmul(matrix_A.astype(np.int32), vector_V.astype(np.int32))
            print("\n矩阵-向量乘积 AV (A @ V) ({} x 1) (dtype: {}):".format(n, result_vector_AV.dtype))
            print(result_vector_AV)
            file_AV_bin = f"{file_prefix}_result_vector_AxV_{n}x1_16bit_binary_no_space.txt" #文件名稍作修改以反映AxV
            try:
                with open(file_AV_bin, 'w') as f_handle:
                    for val_array in result_vector_AV: # result_vector_AV is (n,1)
                        binary_val = to_signed_binary(val_array[0], 16)
                        f_handle.write(binary_val + '\n')
                print(f"结果向量 AV (乘积 AxV, 16位二进制, 无分隔, 每元素一行) 已保存到 {file_AV_bin}")
            except Exception as e:
                print(f"\n保存结果向量 AV 时出错: {e}")


if __name__ == "__main__":
    # 自行定义 n 的值
    n_value = 8
    output_prefix = "my_data"
    
    # --- 用户请求的场景 ---

    # 场景 1: 生成矩阵 A, B 和它们的乘积 C (B@A)
    # print("--- 场景 1: 矩阵与矩阵乘法 (生成 A, B, C=B@A) ---")
    # generate_and_save_data(n_value, file_prefix=f"{output_prefix}_matrix_mult",
    #                        generate_matrix_A=True,
    #                        generate_matrix_B=True,
    #                        perform_matrix_multiplication=True,
    #                        perform_vector_multiplication=False) # 关闭向量乘法

    # 场景 2: 生成矩阵 A, 向量 V 和它们的乘积 AV (A@V)
    print("\n--- 场景 2: 矩阵与向量乘法 (生成 A, V, AV=A@V) ---")
    generate_and_save_data(n_value, file_prefix=f"{output_prefix}_vector_mult",
                           generate_matrix_A=True,
                           generate_matrix_B=False, # 不生成矩阵B
                           perform_matrix_multiplication=False, # 关闭矩阵乘法
                           perform_vector_multiplication=True)
    
    # # --- 其他可选场景示例 ---

    # # 场景 3: 生成所有内容 (矩阵A, B, C=B@A, 向量V, AV=A@V)
    # print("\n--- 场景 3: 生成所有内容 (A, B, C, V, AV) ---")
    # generate_and_save_data(n_value, file_prefix=f"{output_prefix}_all_operations") # 默认所有标志为True
    
    # # 场景 4: 只生成基础矩阵 A 和 B，不进行任何乘法
    # print("\n--- 场景 4: 只生成矩阵 A 和 B (无乘法) ---")
    # generate_and_save_data(n_value, file_prefix=f"{output_prefix}_base_matrices",
    #                        generate_matrix_A=True,
    #                        generate_matrix_B=True,
    #                        perform_matrix_multiplication=False,
    #                        perform_vector_multiplication=False)

    # # 场景 5: 只生成矩阵 A
    # print("\n--- 场景 5: 只生成矩阵 A ---")
    # generate_and_save_data(n_value, file_prefix=f"{output_prefix}_matrix_A_only",
    #                        generate_matrix_A=True,
    #                        generate_matrix_B=False,
    #                        perform_matrix_multiplication=False,
    #                        perform_vector_multiplication=False)
