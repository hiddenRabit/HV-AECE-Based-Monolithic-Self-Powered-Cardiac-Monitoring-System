"""
简化的RR分类器验证：Python vs Verilog（含展平层输出对比）
- Python侧通过forward hook捕获Flatten层输出（24个FP16值）
- 生成的testbench带fp16_to_real函数，可读性输出RTL内部展平层结果（层次引用uut.flatten_out）
- 支持逐位精确比对 + 浮点容差比对
"""

import torch
import torch.nn as nn
import numpy as np

#=============================================================================
# 1. 网络定义（与Verilog完全一致）
#=============================================================================

class RRLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=7, padding=3, bias=False)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=5, padding=2, bias=False)
        self.pool2 = nn.MaxPool1d(3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.adapt_pool = nn.AdaptiveAvgPool1d(3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(24, 2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = self.adapt_pool(x)
        x = self.flatten(x)
        return self.fc(x)

#=============================================================================
# 2. FP16转换函数
#=============================================================================

def float_to_fp16_hex(value):
    """将float32转换为FP16的十六进制表示"""
    fp16 = np.float16(value)
    bits = int(fp16.view(np.uint16))
    return bits

def fp16_hex_to_float(hex_val):
    """将FP16十六进制转换为float32"""
    fp16 = np.uint16(hex_val).view(np.float16)
    return float(fp16)

def format_hex_16bit(value):
    """格式化为Verilog十六进制"""
    value = int(value)
    return f"16'h{value:04X}"

#=============================================================================
# 3. 设置固定权重（与Verilog中的initial块一致）
#=============================================================================

def set_weights_from_verilog(model):
    """设置与Verilog相同的权重"""

    with torch.no_grad():
        # Conv1权重 [3, 1, 7]
        conv1_weights = [
            [1.0, -1.0, 2.0, 3.0, 2.0, -1.0, 1.0],      # 滤波器0
            [0.5, 1.5, -1.5, 0.0, -1.5, 1.5, 0.5],      # 滤波器1
            [-0.5, 1.0, -1.0, 2.0, -1.0, 1.0, -0.5]     # 滤波器2
        ]

        for i in range(3):
            for j in range(7):
                model.conv1.weight.data[i, 0, j] = conv1_weights[i][j]

        # Conv2权重 [8, 3, 5] - 使用简单模式
        for i in range(8):
            for j in range(3):
                for k in range(5):
                    val = ((i + j + k) % 5 - 2) * 0.5
                    model.conv2.weight.data[i, j, k] = val

        # FC权重 [2, 24] 和偏置 [2]
        for i in range(2):
            model.fc.bias.data[i] = 0.1 * (i + 1)
            for j in range(24):
                val = ((i + j) % 7 - 3) * 0.25
                model.fc.weight.data[i, j] = val

    return model

#=============================================================================
# 4. 生成20个测试用例
#=============================================================================

def generate_test_cases():
    """生成20个不同的测试用例"""
    test_cases = []

    # 测试用例1-5：正常窦性心律（不同变异）
    for i in range(5):
        rr_data = [800 + 20 * np.sin(j * np.pi / 5 + i * 0.5) for j in range(20)]
        test_cases.append((rr_data, f"Normal_SR_{i+1}"))

    # 测试用例6-8：心动过速（不同心率）
    for i in range(3):
        base = 500 + i * 30  # 500, 530, 560ms
        rr_data = [base + 10 * np.sin(j * np.pi / 4) for j in range(20)]
        test_cases.append((rr_data, f"Tachycardia_{i+1}"))

    # 测试用例9-11：心动过缓（不同心率）
    for i in range(3):
        base = 1000 + i * 50  # 1000, 1050, 1100ms
        rr_data = [base + 15 * np.sin(j * np.pi / 6) for j in range(20)]
        test_cases.append((rr_data, f"Bradycardia_{i+1}"))

    # 测试用例12-15：房颤（不规则）
    afib_patterns = [
        [650, 920, 580, 1050, 720, 890, 610, 1100, 550, 980,
         780, 640, 1020, 590, 950, 700, 1150, 530, 880, 670],
        [700, 850, 620, 980, 750, 900, 580, 1080, 600, 950,
         720, 680, 1000, 550, 920, 760, 1050, 590, 870, 640],
        [680, 950, 600, 1020, 710, 880, 630, 1120, 570, 990,
         740, 650, 1030, 580, 960, 730, 1080, 560, 890, 660],
        [720, 890, 640, 1000, 730, 910, 590, 1060, 620, 970,
         760, 670, 1010, 600, 930, 710, 1090, 570, 900, 680]
    ]
    for i, pattern in enumerate(afib_patterns):
        test_cases.append((pattern, f"AFib_{i+1}"))

    # 测试用例16-18：混合模式
    test_cases.append(([850, 780, 920, 810, 880, 790, 900, 820, 860, 830,
                        870, 800, 910, 840, 890, 810, 880, 820, 900, 850], "Mixed_1"))
    test_cases.append(([750, 950, 680, 1020, 790, 880, 720, 980, 810, 900,
                        760, 940, 700, 1000, 780, 920, 740, 960, 800, 890], "Mixed_2"))
    test_cases.append(([600, 850, 1100, 700, 950, 650, 1000, 750, 900, 800,
                        1050, 680, 980, 720, 920, 660, 1010, 780, 880, 700], "Mixed_3"))

    # 测试用例19-20：边界情况
    test_cases.append(([300] * 20, "All_Min_300ms"))  # 全部最小值
    test_cases.append(([2000] * 20, "All_Max_2000ms"))  # 全部最大值

    return test_cases

#=============================================================================
# 5. Python前向传播（FP16精度，同时捕获展平层输出）
#=============================================================================

def forward_fp16(model, rr_data):
    """使用FP16精度进行前向传播，同时捕获Flatten层输出"""

    model.eval()
    model.half()  # 转换为FP16

    # 存储展平层输出
    flatten_output = None
    def hook_fn(module, input, output):
        nonlocal flatten_output
        flatten_output = output.detach().cpu().numpy().copy()

    # 注册hook到flatten层
    handle = model.flatten.register_forward_hook(hook_fn)

    with torch.no_grad():
        x = torch.tensor(rr_data, dtype=torch.float16).unsqueeze(0).unsqueeze(0)

        # 前向传播
        logits = model(x)
        handle.remove()

        # 获取logits结果
        logits_np = logits.squeeze().cpu().numpy()
        prediction = 0 if logits_np[0] > logits_np[1] else 1
        logits_hex = [float_to_fp16_hex(x) for x in logits_np]

        # 展平层输出：24个FP16值（flatten_output为float16 numpy数组）
        flat_np = flatten_output.squeeze().astype(np.float16)
        assert flat_np.size == 24, f"flatten size error: {flat_np.size}"
        flat_hex = [float_to_fp16_hex(float(v)) for v in flat_np]

    return {
        'logits_float': [float(x) for x in logits_np],
        'logits_hex': logits_hex,
        'prediction': prediction,
        'flatten_float': [float(v) for v in flat_np],   # 24个float值
        'flatten_hex': flat_hex                          # 24个16bit hex
    }

#=============================================================================
# 6. 生成权重初始化代码
#=============================================================================

def generate_weight_init_code(model):
    """生成权重初始化的Verilog代码"""
    lines = []

    with torch.no_grad():
        # Conv1权重
        lines.append("    // Conv1 weights [3][7]")
        for i in range(3):
            for j in range(7):
                val = model.conv1.weight.data[i, 0, j].item()
                hex_val = float_to_fp16_hex(val)
                lines.append(f"    conv1_weight[{i}][{j}] = {format_hex_16bit(hex_val)}; // {val:.6f}")

        # Conv2权重
        lines.append("    // Conv2 weights [8][3][5]")
        for i in range(8):
            for j in range(3):
                for k in range(5):
                    val = model.conv2.weight.data[i, j, k].item()
                    hex_val = float_to_fp16_hex(val)
                    lines.append(f"    conv2_weight[{i}][{j}][{k}] = {format_hex_16bit(hex_val)}; // {val:.6f}")

        # FC权重和偏置
        lines.append("    // FC bias [2]")
        for i in range(2):
            val = model.fc.bias.data[i].item()
            hex_val = float_to_fp16_hex(val)
            lines.append(f"    fc_bias[{i}] = {format_hex_16bit(hex_val)}; // {val:.6f}")

        lines.append("    // FC weights [2][24]")
        for i in range(2):
            for j in range(24):
                val = model.fc.weight.data[i, j].item()
                hex_val = float_to_fp16_hex(val)
                lines.append(f"    fc_weight[{i}][{j}] = {format_hex_16bit(hex_val)}; // {val:.6f}")

    return '\n'.join(lines)

#=============================================================================
# 7. 生成Verilog测试平台（带fp16_to_real + 展平层内部信号读取）
#=============================================================================

def generate_complete_testbench(model, test_cases, results, output_file='testbench_flatten.sv'):
    """生成完整的Verilog测试平台：
    - 带fp16_to_real函数，把FP16十六进制转成可读浮点数
    - 通过层次引用uut.flatten_out读取RTL内部展平层结果并输出
    - 逐位精确比对 + 容差比对
    """

    # 生成权重初始化代码
    weight_init = generate_weight_init_code(model)

    tb = """//=============================================================================
// RR分类器完整测试平台（含展平层输出验证）
// 通过层次引用 uut.flatten_out 读取RTL内部展平层结果，无需修改DUT
//=============================================================================

`timescale 1ns/1ps

module testbench_flatten;

//=============================================================================
// 信号定义
//=============================================================================
reg clk;
reg rst_n;
reg start;
reg [15:0] rr_data [0:19];
wire done;
wire prediction;
wire [15:0] logits_out [0:1];

//=============================================================================
// FP16 -> Real 转换函数（用于可读性输出和容差比较）
//=============================================================================
function real pow2;
    input integer n;
    real r;
    integer i;
    begin
        r = 1.0;
        if (n >= 0) begin
            for (i = 0; i < n; i = i + 1) r = r * 2.0;
        end else begin
            for (i = 0; i < -n; i = i + 1) r = r / 2.0;
        end
        pow2 = r;
    end
endfunction

function real fp16_to_real;
    input [15:0] h;
    reg        sign;
    reg [4:0]  exp;
    reg [9:0]  mant;
    real       frac;
    real       result;
    integer    i;
    begin
        sign = h[15];
        exp  = h[14:10];
        mant = h[9:0];
        // 尾数小数部分
        frac = 0.0;
        for (i = 0; i < 10; i = i + 1)
            if (mant[i]) frac = frac + pow2(i - 10);
        if (exp == 5'b00000) begin
            // 次正规数（含0）
            result = frac * pow2(-14);
        end else if (exp == 5'b11111) begin
            // Inf / NaN（简化为大数/0）
            result = (mant != 0) ? 0.0 : frac * pow2(16);
        end else begin
            // 正规数：1.f * 2^(exp-15)
            result = (1.0 + frac) * pow2(exp - 15);
        end
        fp16_to_real = sign ? -result : result;
    end
endfunction

// 绝对值函数
function real real_abs;
    input real x;
    begin
        real_abs = (x < 0.0) ? -x : x;
    end
endfunction

//=============================================================================
// 权重存储（与DUT中的权重相同）
//=============================================================================
reg [15:0] conv1_weight [0:2][0:6];
reg [15:0] conv2_weight [0:7][0:2][0:4];
reg [15:0] fc_weight [0:1][0:23];
reg [15:0] fc_bias [0:1];

// 权重初始化
initial begin
"""
    tb += weight_init
    tb += """
end

//=============================================================================
// 实例化被测模块（RTL无需任何修改）
//=============================================================================
rr_classifier_fp16 uut(
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .rr_data(rr_data),
    .done(done),
    .prediction(prediction),
    .logits_out(logits_out)
);

//=============================================================================
// 时钟生成
//=============================================================================
initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100MHz
end

//=============================================================================
// 测试数据存储
//=============================================================================
reg [15:0] test_data [0:19][0:19];        // 20个测试用例，每个20个数据
reg [19:0] expected_results;              // 20个期望分类结果
reg [15:0] expected_flatten [0:19][0:23]; // 20个用例的期望展平层输出（24个FP16）
integer test_num;
integer errors;
integer i;

//=============================================================================
// 测试任务
//=============================================================================
task run_test;
    input integer test_id;
    integer k;
    integer exact_match;      // 逐位精确匹配数
    integer tol_match;        // 容差匹配数
    real    rtl_r, py_r, diff, rel;
    begin
        // 加载测试数据
        for (i = 0; i < 20; i = i + 1) begin
            rr_data[i] = test_data[test_id][i];
        end

        // 启动计算
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // 等待完成
        wait(done);

        // ---- 分类结果比对 ----
        $display("------------------------------------------------------------");
        $display("Test %0d: prediction=%b, expected=%b, logits=[%h, %h] = [%f, %f]",
                 test_id, prediction, expected_results[test_id],
                 logits_out[0], logits_out[1],
                 fp16_to_real(logits_out[0]), fp16_to_real(logits_out[1]));

        if (prediction !== expected_results[test_id]) begin
            $display("  *** ERROR: Test %0d classification failed!", test_id);
            errors = errors + 1;
        end

        // ---- 展平层结果：读取RTL内部信号并逐点输出 ----
        exact_match = 0;
        tol_match   = 0;
        $display("  Flatten layer (uut.flatten_out):");
        $display("    idx |  ch.pt |      RTL(hex)   RTL(real)    |      PY(hex)    PY(real)   | result");
        for (k = 0; k < 24; k = k + 1) begin
            rtl_r = fp16_to_real(uut.flatten_out[k]);
            py_r  = fp16_to_real(expected_flatten[test_id][k]);
            diff  = real_abs(rtl_r - py_r);
            rel   = (real_abs(py_r) > 0.000001) ? diff / real_abs(py_r) : diff;

            if (uut.flatten_out[k] === expected_flatten[test_id][k]) begin
                exact_match = exact_match + 1;
                tol_match   = tol_match + 1;
                $display("    %2d  |  %0d.%0d   |  %h  %10.4f  |  %h  %10.4f  | EXACT",
                         k, k/3, k%3,
                         uut.flatten_out[k], rtl_r,
                         expected_flatten[test_id][k], py_r);
            end else if (rel < 0.01) begin
                tol_match = tol_match + 1;
                $display("    %2d  |  %0d.%0d   |  %h  %10.4f  |  %h  %10.4f  | TOL_OK (rel=%f)",
                         k, k/3, k%3,
                         uut.flatten_out[k], rtl_r,
                         expected_flatten[test_id][k], py_r, rel);
            end else begin
                $display("    %2d  |  %0d.%0d   |  %h  %10.4f  |  %h  %10.4f  | ***MISMATCH*** (diff=%f)",
                         k, k/3, k%3,
                         uut.flatten_out[k], rtl_r,
                         expected_flatten[test_id][k], py_r, diff);
            end
        end

        $display("  Flatten summary: exact %0d/24, tolerance(1%%) %0d/24",
                 exact_match, tol_match);

        if (tol_match < 24) begin
            $display("  *** ERROR: Test %0d flatten tolerance check failed!", test_id);
            errors = errors + 1;
        end

        @(posedge clk);
    end
endtask

//=============================================================================
// 主测试流程
//=============================================================================
initial begin
    // 初始化
    rst_n = 0;
    start = 0;
    errors = 0;
    test_num = 0;
    expected_results = 0;

    // 复位
    #100;
    rst_n = 1;
    #50;

    $display("============================================================");
    $display("RR分类器测试 - 20个测试用例（含展平层内部结果验证）");
    $display("============================================================");

"""
    # 添加20个测试用例的数据、期望分类结果和期望展平层输出
    for idx, ((rr_data, name), result) in enumerate(zip(test_cases, results)):
        tb += f"    // Test {idx}: {name}\n"
        tb += f"    expected_results[{idx}] = 1'b{result['prediction']};\n"

        for j, val in enumerate(rr_data):
            hex_val = float_to_fp16_hex(val)
            tb += f"    test_data[{idx}][{j}] = {format_hex_16bit(hex_val)}; // {val:.1f}\n"

        # 期望展平层输出
        tb += f"    // 期望展平层输出 (Flatten 24 values)\n"
        for j, hex_val in enumerate(result['flatten_hex']):
            tb += f"    expected_flatten[{idx}][{j}] = {format_hex_16bit(hex_val)}; // {result['flatten_float'][j]:.6f}\n"

        tb += "\n"

    # 添加测试执行代码
    tb += """
    // 执行所有测试
    for (test_num = 0; test_num < 20; test_num = test_num + 1) begin
        run_test(test_num);
    end

    // 测试总结
    $display("");
    $display("============================================================");
    $display("测试总结:");
    $display("  总测试数: 20");
    $display("  错误数: %0d", errors);
    if (errors == 0)
        $display("  结果: 所有测试通过!");
    else
        $display("  结果: 有%0d个测试失败!", errors);
    $display("============================================================");

    #1000;
    $finish;
end

//=============================================================================
// 波形输出（包含UUT内部所有信号：flatten_out / adaptpool_out / pool2_out等）
//=============================================================================
initial begin
    $dumpfile("testbench_flatten.vcd");
    $dumpvars(0, testbench_flatten);
end

//=============================================================================
// 超时保护
//=============================================================================
initial begin
    #1000000;
    $display("超时错误：测试未在预期时间内完成");
    $finish;
end

endmodule
"""

    with open(output_file, 'w') as f:
        f.write(tb)

    print(f"完整测试平台已生成: {output_file}")

#=============================================================================
# 8. 主验证流程
#=============================================================================

def main():
    print("="*60)
    print("RR分类器验证：Python (FP16) vs Verilog [含展平层输出]")
    print("="*60)

    # 1. 创建模型并设置权重
    print("\n1. 创建模型并设置权重...")
    model = RRLite()
    model = set_weights_from_verilog(model)
    print("   权重设置完成")

    # 2. 生成测试用例
    print("\n2. 生成20个测试用例...")
    test_cases = generate_test_cases()
    print(f"   生成了{len(test_cases)}个测试用例")

    # 3. Python前向传播
    print("\n3. Python前向传播（FP16，捕获展平层输出）...")
    print("="*60)
    print("Python预测结果:")
    print("-"*60)

    results = []
    for idx, (rr_data, name) in enumerate(test_cases):
        result = forward_fp16(model, rr_data)
        results.append(result)
        label = "SR" if result['prediction'] == 0 else "NON-SR"
        print(f"  Test {idx+1:2d} ({name:20s}): prediction={result['prediction']} ({label}), "
              f"logits=[{result['logits_float'][0]:.4f}, {result['logits_float'][1]:.4f}]")

    # 打印第一个测试用例的展平层输出作为示例
    print("-"*60)
    print("示例 - Test 1 展平层输出（24个值）:")
    for j, v in enumerate(results[0]['flatten_float']):
        print(f"  flatten[{j:2d}] (ch{j//3}.pt{j%3}) = {v:10.4f}  ({format_hex_16bit(results[0]['flatten_hex'][j])})")

    print("="*60)

    # 4. 生成Verilog测试平台
    print("\n4. 生成Verilog测试平台（带fp16_to_real和展平层比对）...")
    generate_complete_testbench(model, test_cases, results, 'testbench_flatten.sv')

    # 5. 输出总结
    print("\n" + "="*60)
    print("验证文件生成完成！")
    print("="*60)
    print("\n生成的文件：")
    print("  - testbench_flatten.sv : 完整测试平台（含权重、20个测试用例、展平层期望数据）")
    print("\n运行Verilog仿真：")
    print("  iverilog -o test rr_classifier_fp16.sv testbench_flatten.sv")
    print("  vvp test")
    print("\n说明：")
    print("  - tb通过层次引用 uut.flatten_out 读取RTL内部展平层结果（RTL无需修改）")
    print("  - 每个测试打印24个展平层值的 RTL/PY 对比（hex + 十进制浮点）")
    print("  - 提供逐位精确比对和1%容差比对两种统计")

if __name__ == "__main__":
    main()
