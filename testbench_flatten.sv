//=============================================================================
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
    // Conv1 weights [3][7]
    conv1_weight[0][0] = 16'h3C00; // 1.000000
    conv1_weight[0][1] = 16'hBC00; // -1.000000
    conv1_weight[0][2] = 16'h4000; // 2.000000
    conv1_weight[0][3] = 16'h4200; // 3.000000
    conv1_weight[0][4] = 16'h4000; // 2.000000
    conv1_weight[0][5] = 16'hBC00; // -1.000000
    conv1_weight[0][6] = 16'h3C00; // 1.000000
    conv1_weight[1][0] = 16'h3800; // 0.500000
    conv1_weight[1][1] = 16'h3E00; // 1.500000
    conv1_weight[1][2] = 16'hBE00; // -1.500000
    conv1_weight[1][3] = 16'h0000; // 0.000000
    conv1_weight[1][4] = 16'hBE00; // -1.500000
    conv1_weight[1][5] = 16'h3E00; // 1.500000
    conv1_weight[1][6] = 16'h3800; // 0.500000
    conv1_weight[2][0] = 16'hB800; // -0.500000
    conv1_weight[2][1] = 16'h3C00; // 1.000000
    conv1_weight[2][2] = 16'hBC00; // -1.000000
    conv1_weight[2][3] = 16'h4000; // 2.000000
    conv1_weight[2][4] = 16'hBC00; // -1.000000
    conv1_weight[2][5] = 16'h3C00; // 1.000000
    conv1_weight[2][6] = 16'hB800; // -0.500000
    // Conv2 weights [8][3][5]
    conv2_weight[0][0][0] = 16'hBC00; // -1.000000
    conv2_weight[0][0][1] = 16'hB800; // -0.500000
    conv2_weight[0][0][2] = 16'h0000; // 0.000000
    conv2_weight[0][0][3] = 16'h3800; // 0.500000
    conv2_weight[0][0][4] = 16'h3C00; // 1.000000
    conv2_weight[0][1][0] = 16'hB800; // -0.500000
    conv2_weight[0][1][1] = 16'h0000; // 0.000000
    conv2_weight[0][1][2] = 16'h3800; // 0.500000
    conv2_weight[0][1][3] = 16'h3C00; // 1.000000
    conv2_weight[0][1][4] = 16'hBC00; // -1.000000
    conv2_weight[0][2][0] = 16'h0000; // 0.000000
    conv2_weight[0][2][1] = 16'h3800; // 0.500000
    conv2_weight[0][2][2] = 16'h3C00; // 1.000000
    conv2_weight[0][2][3] = 16'hBC00; // -1.000000
    conv2_weight[0][2][4] = 16'hB800; // -0.500000
    conv2_weight[1][0][0] = 16'hB800; // -0.500000
    conv2_weight[1][0][1] = 16'h0000; // 0.000000
    conv2_weight[1][0][2] = 16'h3800; // 0.500000
    conv2_weight[1][0][3] = 16'h3C00; // 1.000000
    conv2_weight[1][0][4] = 16'hBC00; // -1.000000
    conv2_weight[1][1][0] = 16'h0000; // 0.000000
    conv2_weight[1][1][1] = 16'h3800; // 0.500000
    conv2_weight[1][1][2] = 16'h3C00; // 1.000000
    conv2_weight[1][1][3] = 16'hBC00; // -1.000000
    conv2_weight[1][1][4] = 16'hB800; // -0.500000
    conv2_weight[1][2][0] = 16'h3800; // 0.500000
    conv2_weight[1][2][1] = 16'h3C00; // 1.000000
    conv2_weight[1][2][2] = 16'hBC00; // -1.000000
    conv2_weight[1][2][3] = 16'hB800; // -0.500000
    conv2_weight[1][2][4] = 16'h0000; // 0.000000
    conv2_weight[2][0][0] = 16'h0000; // 0.000000
    conv2_weight[2][0][1] = 16'h3800; // 0.500000
    conv2_weight[2][0][2] = 16'h3C00; // 1.000000
    conv2_weight[2][0][3] = 16'hBC00; // -1.000000
    conv2_weight[2][0][4] = 16'hB800; // -0.500000
    conv2_weight[2][1][0] = 16'h3800; // 0.500000
    conv2_weight[2][1][1] = 16'h3C00; // 1.000000
    conv2_weight[2][1][2] = 16'hBC00; // -1.000000
    conv2_weight[2][1][3] = 16'hB800; // -0.500000
    conv2_weight[2][1][4] = 16'h0000; // 0.000000
    conv2_weight[2][2][0] = 16'h3C00; // 1.000000
    conv2_weight[2][2][1] = 16'hBC00; // -1.000000
    conv2_weight[2][2][2] = 16'hB800; // -0.500000
    conv2_weight[2][2][3] = 16'h0000; // 0.000000
    conv2_weight[2][2][4] = 16'h3800; // 0.500000
    conv2_weight[3][0][0] = 16'h3800; // 0.500000
    conv2_weight[3][0][1] = 16'h3C00; // 1.000000
    conv2_weight[3][0][2] = 16'hBC00; // -1.000000
    conv2_weight[3][0][3] = 16'hB800; // -0.500000
    conv2_weight[3][0][4] = 16'h0000; // 0.000000
    conv2_weight[3][1][0] = 16'h3C00; // 1.000000
    conv2_weight[3][1][1] = 16'hBC00; // -1.000000
    conv2_weight[3][1][2] = 16'hB800; // -0.500000
    conv2_weight[3][1][3] = 16'h0000; // 0.000000
    conv2_weight[3][1][4] = 16'h3800; // 0.500000
    conv2_weight[3][2][0] = 16'hBC00; // -1.000000
    conv2_weight[3][2][1] = 16'hB800; // -0.500000
    conv2_weight[3][2][2] = 16'h0000; // 0.000000
    conv2_weight[3][2][3] = 16'h3800; // 0.500000
    conv2_weight[3][2][4] = 16'h3C00; // 1.000000
    conv2_weight[4][0][0] = 16'h3C00; // 1.000000
    conv2_weight[4][0][1] = 16'hBC00; // -1.000000
    conv2_weight[4][0][2] = 16'hB800; // -0.500000
    conv2_weight[4][0][3] = 16'h0000; // 0.000000
    conv2_weight[4][0][4] = 16'h3800; // 0.500000
    conv2_weight[4][1][0] = 16'hBC00; // -1.000000
    conv2_weight[4][1][1] = 16'hB800; // -0.500000
    conv2_weight[4][1][2] = 16'h0000; // 0.000000
    conv2_weight[4][1][3] = 16'h3800; // 0.500000
    conv2_weight[4][1][4] = 16'h3C00; // 1.000000
    conv2_weight[4][2][0] = 16'hB800; // -0.500000
    conv2_weight[4][2][1] = 16'h0000; // 0.000000
    conv2_weight[4][2][2] = 16'h3800; // 0.500000
    conv2_weight[4][2][3] = 16'h3C00; // 1.000000
    conv2_weight[4][2][4] = 16'hBC00; // -1.000000
    conv2_weight[5][0][0] = 16'hBC00; // -1.000000
    conv2_weight[5][0][1] = 16'hB800; // -0.500000
    conv2_weight[5][0][2] = 16'h0000; // 0.000000
    conv2_weight[5][0][3] = 16'h3800; // 0.500000
    conv2_weight[5][0][4] = 16'h3C00; // 1.000000
    conv2_weight[5][1][0] = 16'hB800; // -0.500000
    conv2_weight[5][1][1] = 16'h0000; // 0.000000
    conv2_weight[5][1][2] = 16'h3800; // 0.500000
    conv2_weight[5][1][3] = 16'h3C00; // 1.000000
    conv2_weight[5][1][4] = 16'hBC00; // -1.000000
    conv2_weight[5][2][0] = 16'h0000; // 0.000000
    conv2_weight[5][2][1] = 16'h3800; // 0.500000
    conv2_weight[5][2][2] = 16'h3C00; // 1.000000
    conv2_weight[5][2][3] = 16'hBC00; // -1.000000
    conv2_weight[5][2][4] = 16'hB800; // -0.500000
    conv2_weight[6][0][0] = 16'hB800; // -0.500000
    conv2_weight[6][0][1] = 16'h0000; // 0.000000
    conv2_weight[6][0][2] = 16'h3800; // 0.500000
    conv2_weight[6][0][3] = 16'h3C00; // 1.000000
    conv2_weight[6][0][4] = 16'hBC00; // -1.000000
    conv2_weight[6][1][0] = 16'h0000; // 0.000000
    conv2_weight[6][1][1] = 16'h3800; // 0.500000
    conv2_weight[6][1][2] = 16'h3C00; // 1.000000
    conv2_weight[6][1][3] = 16'hBC00; // -1.000000
    conv2_weight[6][1][4] = 16'hB800; // -0.500000
    conv2_weight[6][2][0] = 16'h3800; // 0.500000
    conv2_weight[6][2][1] = 16'h3C00; // 1.000000
    conv2_weight[6][2][2] = 16'hBC00; // -1.000000
    conv2_weight[6][2][3] = 16'hB800; // -0.500000
    conv2_weight[6][2][4] = 16'h0000; // 0.000000
    conv2_weight[7][0][0] = 16'h0000; // 0.000000
    conv2_weight[7][0][1] = 16'h3800; // 0.500000
    conv2_weight[7][0][2] = 16'h3C00; // 1.000000
    conv2_weight[7][0][3] = 16'hBC00; // -1.000000
    conv2_weight[7][0][4] = 16'hB800; // -0.500000
    conv2_weight[7][1][0] = 16'h3800; // 0.500000
    conv2_weight[7][1][1] = 16'h3C00; // 1.000000
    conv2_weight[7][1][2] = 16'hBC00; // -1.000000
    conv2_weight[7][1][3] = 16'hB800; // -0.500000
    conv2_weight[7][1][4] = 16'h0000; // 0.000000
    conv2_weight[7][2][0] = 16'h3C00; // 1.000000
    conv2_weight[7][2][1] = 16'hBC00; // -1.000000
    conv2_weight[7][2][2] = 16'hB800; // -0.500000
    conv2_weight[7][2][3] = 16'h0000; // 0.000000
    conv2_weight[7][2][4] = 16'h3800; // 0.500000
    // FC bias [2]
    fc_bias[0] = 16'h2E66; // 0.099976
    fc_bias[1] = 16'h3266; // 0.199951
    // FC weights [2][24]
    fc_weight[0][0] = 16'hBA00; // -0.750000
    fc_weight[0][1] = 16'hB800; // -0.500000
    fc_weight[0][2] = 16'hB400; // -0.250000
    fc_weight[0][3] = 16'h0000; // 0.000000
    fc_weight[0][4] = 16'h3400; // 0.250000
    fc_weight[0][5] = 16'h3800; // 0.500000
    fc_weight[0][6] = 16'h3A00; // 0.750000
    fc_weight[0][7] = 16'hBA00; // -0.750000
    fc_weight[0][8] = 16'hB800; // -0.500000
    fc_weight[0][9] = 16'hB400; // -0.250000
    fc_weight[0][10] = 16'h0000; // 0.000000
    fc_weight[0][11] = 16'h3400; // 0.250000
    fc_weight[0][12] = 16'h3800; // 0.500000
    fc_weight[0][13] = 16'h3A00; // 0.750000
    fc_weight[0][14] = 16'hBA00; // -0.750000
    fc_weight[0][15] = 16'hB800; // -0.500000
    fc_weight[0][16] = 16'hB400; // -0.250000
    fc_weight[0][17] = 16'h0000; // 0.000000
    fc_weight[0][18] = 16'h3400; // 0.250000
    fc_weight[0][19] = 16'h3800; // 0.500000
    fc_weight[0][20] = 16'h3A00; // 0.750000
    fc_weight[0][21] = 16'hBA00; // -0.750000
    fc_weight[0][22] = 16'hB800; // -0.500000
    fc_weight[0][23] = 16'hB400; // -0.250000
    fc_weight[1][0] = 16'hB800; // -0.500000
    fc_weight[1][1] = 16'hB400; // -0.250000
    fc_weight[1][2] = 16'h0000; // 0.000000
    fc_weight[1][3] = 16'h3400; // 0.250000
    fc_weight[1][4] = 16'h3800; // 0.500000
    fc_weight[1][5] = 16'h3A00; // 0.750000
    fc_weight[1][6] = 16'hBA00; // -0.750000
    fc_weight[1][7] = 16'hB800; // -0.500000
    fc_weight[1][8] = 16'hB400; // -0.250000
    fc_weight[1][9] = 16'h0000; // 0.000000
    fc_weight[1][10] = 16'h3400; // 0.250000
    fc_weight[1][11] = 16'h3800; // 0.500000
    fc_weight[1][12] = 16'h3A00; // 0.750000
    fc_weight[1][13] = 16'hBA00; // -0.750000
    fc_weight[1][14] = 16'hB800; // -0.500000
    fc_weight[1][15] = 16'hB400; // -0.250000
    fc_weight[1][16] = 16'h0000; // 0.000000
    fc_weight[1][17] = 16'h3400; // 0.250000
    fc_weight[1][18] = 16'h3800; // 0.500000
    fc_weight[1][19] = 16'h3A00; // 0.750000
    fc_weight[1][20] = 16'hBA00; // -0.750000
    fc_weight[1][21] = 16'hB800; // -0.500000
    fc_weight[1][22] = 16'hB400; // -0.250000
    fc_weight[1][23] = 16'h0000; // 0.000000
    for (integer o = 0; o < 8; o = o + 1)
        for (integer i = 0; i < 3; i = i + 1)
            for (integer k = 0; k < 5; k = k + 1)
                if (^conv2_weight[o][i][k] === 1'bx)
                    $display("X FOUND: conv2_weight[%0d][%0d][%0d]", o, i, k);
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

    // Test 0: Normal_SR_1
    expected_results[0] = 1'b1;
    test_data[0][0] = 16'h6240; // 800.0
    test_data[0][1] = 16'h6258; // 811.8
    test_data[0][2] = 16'h6266; // 819.0
    test_data[0][3] = 16'h6266; // 819.0
    test_data[0][4] = 16'h6258; // 811.8
    test_data[0][5] = 16'h6240; // 800.0
    test_data[0][6] = 16'h6228; // 788.2
    test_data[0][7] = 16'h621A; // 781.0
    test_data[0][8] = 16'h621A; // 781.0
    test_data[0][9] = 16'h6228; // 788.2
    test_data[0][10] = 16'h6240; // 800.0
    test_data[0][11] = 16'h6258; // 811.8
    test_data[0][12] = 16'h6266; // 819.0
    test_data[0][13] = 16'h6266; // 819.0
    test_data[0][14] = 16'h6258; // 811.8
    test_data[0][15] = 16'h6240; // 800.0
    test_data[0][16] = 16'h6228; // 788.2
    test_data[0][17] = 16'h621A; // 781.0
    test_data[0][18] = 16'h621A; // 781.0
    test_data[0][19] = 16'h6228; // 788.2
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[0][0] = 16'h6F40; // 7424.000000
    expected_flatten[0][1] = 16'h686B; // 2262.000000
    expected_flatten[0][2] = 16'h5344; // 58.125000
    expected_flatten[0][3] = 16'h686E; // 2268.000000
    expected_flatten[0][4] = 16'h6264; // 818.000000
    expected_flatten[0][5] = 16'h69B1; // 2914.000000
    expected_flatten[0][6] = 16'h5A6B; // 205.375000
    expected_flatten[0][7] = 16'h5E14; // 389.000000
    expected_flatten[0][8] = 16'h6C24; // 4240.000000
    expected_flatten[0][9] = 16'h0000; // 0.000000
    expected_flatten[0][10] = 16'h58BD; // 151.625000
    expected_flatten[0][11] = 16'h62C9; // 868.500000
    expected_flatten[0][12] = 16'h649D; // 1181.000000
    expected_flatten[0][13] = 16'h567D; // 103.812500
    expected_flatten[0][14] = 16'h49D0; // 11.625000
    expected_flatten[0][15] = 16'h6F40; // 7424.000000
    expected_flatten[0][16] = 16'h686B; // 2262.000000
    expected_flatten[0][17] = 16'h5344; // 58.125000
    expected_flatten[0][18] = 16'h686E; // 2268.000000
    expected_flatten[0][19] = 16'h6264; // 818.000000
    expected_flatten[0][20] = 16'h69B1; // 2914.000000
    expected_flatten[0][21] = 16'h5A6B; // 205.375000
    expected_flatten[0][22] = 16'h5E14; // 389.000000
    expected_flatten[0][23] = 16'h6C24; // 4240.000000

    // Test 1: Normal_SR_2
    expected_results[1] = 1'b1;
    test_data[1][0] = 16'h6253; // 809.6
    test_data[1][1] = 16'h6264; // 818.1
    test_data[1][2] = 16'h6267; // 819.7
    test_data[1][3] = 16'h625B; // 813.7
    test_data[1][4] = 16'h6245; // 802.6
    test_data[1][5] = 16'h622D; // 790.4
    test_data[1][6] = 16'h621C; // 781.9
    test_data[1][7] = 16'h6219; // 780.3
    test_data[1][8] = 16'h6225; // 786.3
    test_data[1][9] = 16'h623B; // 797.4
    test_data[1][10] = 16'h6253; // 809.6
    test_data[1][11] = 16'h6264; // 818.1
    test_data[1][12] = 16'h6267; // 819.7
    test_data[1][13] = 16'h625B; // 813.7
    test_data[1][14] = 16'h6245; // 802.6
    test_data[1][15] = 16'h622D; // 790.4
    test_data[1][16] = 16'h621C; // 781.9
    test_data[1][17] = 16'h6219; // 780.3
    test_data[1][18] = 16'h6225; // 786.3
    test_data[1][19] = 16'h623B; // 797.4
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[1][0] = 16'h6F38; // 7392.000000
    expected_flatten[1][1] = 16'h6861; // 2242.000000
    expected_flatten[1][2] = 16'h524C; // 50.375000
    expected_flatten[1][3] = 16'h687A; // 2292.000000
    expected_flatten[1][4] = 16'h6277; // 827.500000
    expected_flatten[1][5] = 16'h69AE; // 2908.000000
    expected_flatten[1][6] = 16'h5929; // 165.125000
    expected_flatten[1][7] = 16'h5D2F; // 331.750000
    expected_flatten[1][8] = 16'h6C25; // 4244.000000
    expected_flatten[1][9] = 16'h0000; // 0.000000
    expected_flatten[1][10] = 16'h5908; // 161.000000
    expected_flatten[1][11] = 16'h62AB; // 853.500000
    expected_flatten[1][12] = 16'h64B5; // 1205.000000
    expected_flatten[1][13] = 16'h5670; // 103.000000
    expected_flatten[1][14] = 16'h0000; // 0.000000
    expected_flatten[1][15] = 16'h6F38; // 7392.000000
    expected_flatten[1][16] = 16'h6861; // 2242.000000
    expected_flatten[1][17] = 16'h524C; // 50.375000
    expected_flatten[1][18] = 16'h687A; // 2292.000000
    expected_flatten[1][19] = 16'h6277; // 827.500000
    expected_flatten[1][20] = 16'h69AE; // 2908.000000
    expected_flatten[1][21] = 16'h5929; // 165.125000
    expected_flatten[1][22] = 16'h5D2F; // 331.750000
    expected_flatten[1][23] = 16'h6C25; // 4244.000000

    // Test 2: Normal_SR_3
    expected_results[2] = 1'b1;
    test_data[2][0] = 16'h6262; // 816.8
    test_data[2][1] = 16'h6268; // 820.0
    test_data[2][2] = 16'h625F; // 815.5
    test_data[2][3] = 16'h624A; // 805.1
    test_data[2][4] = 16'h6231; // 792.7
    test_data[2][5] = 16'h621E; // 783.2
    test_data[2][6] = 16'h6218; // 780.0
    test_data[2][7] = 16'h6221; // 784.5
    test_data[2][8] = 16'h6236; // 794.9
    test_data[2][9] = 16'h624F; // 807.3
    test_data[2][10] = 16'h6262; // 816.8
    test_data[2][11] = 16'h6268; // 820.0
    test_data[2][12] = 16'h625F; // 815.5
    test_data[2][13] = 16'h624A; // 805.1
    test_data[2][14] = 16'h6231; // 792.7
    test_data[2][15] = 16'h621E; // 783.2
    test_data[2][16] = 16'h6218; // 780.0
    test_data[2][17] = 16'h6221; // 784.5
    test_data[2][18] = 16'h6236; // 794.9
    test_data[2][19] = 16'h624F; // 807.3
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[2][0] = 16'h6F34; // 7376.000000
    expected_flatten[2][1] = 16'h685D; // 2234.000000
    expected_flatten[2][2] = 16'h5050; // 34.500000
    expected_flatten[2][3] = 16'h687C; // 2296.000000
    expected_flatten[2][4] = 16'h6299; // 844.500000
    expected_flatten[2][5] = 16'h69B8; // 2928.000000
    expected_flatten[2][6] = 16'h582C; // 133.500000
    expected_flatten[2][7] = 16'h5C7C; // 287.000000
    expected_flatten[2][8] = 16'h6C26; // 4248.000000
    expected_flatten[2][9] = 16'h0000; // 0.000000
    expected_flatten[2][10] = 16'h597D; // 175.625000
    expected_flatten[2][11] = 16'h6284; // 834.000000
    expected_flatten[2][12] = 16'h64C1; // 1217.000000
    expected_flatten[2][13] = 16'h5608; // 96.500000
    expected_flatten[2][14] = 16'h0000; // 0.000000
    expected_flatten[2][15] = 16'h6F34; // 7376.000000
    expected_flatten[2][16] = 16'h685D; // 2234.000000
    expected_flatten[2][17] = 16'h5050; // 34.500000
    expected_flatten[2][18] = 16'h687C; // 2296.000000
    expected_flatten[2][19] = 16'h6299; // 844.500000
    expected_flatten[2][20] = 16'h69B8; // 2928.000000
    expected_flatten[2][21] = 16'h582C; // 133.500000
    expected_flatten[2][22] = 16'h5C7C; // 287.000000
    expected_flatten[2][23] = 16'h6C26; // 4248.000000

    // Test 3: Normal_SR_4
    expected_results[3] = 1'b1;
    test_data[3][0] = 16'h6268; // 819.9
    test_data[3][1] = 16'h6262; // 817.0
    test_data[3][2] = 16'h624F; // 807.5
    test_data[3][3] = 16'h6236; // 795.2
    test_data[3][4] = 16'h6221; // 784.7
    test_data[3][5] = 16'h6218; // 780.1
    test_data[3][6] = 16'h621E; // 783.0
    test_data[3][7] = 16'h6231; // 792.5
    test_data[3][8] = 16'h624A; // 804.8
    test_data[3][9] = 16'h625F; // 815.3
    test_data[3][10] = 16'h6268; // 819.9
    test_data[3][11] = 16'h6262; // 817.0
    test_data[3][12] = 16'h624F; // 807.5
    test_data[3][13] = 16'h6236; // 795.2
    test_data[3][14] = 16'h6221; // 784.7
    test_data[3][15] = 16'h6218; // 780.1
    test_data[3][16] = 16'h621E; // 783.0
    test_data[3][17] = 16'h6231; // 792.5
    test_data[3][18] = 16'h624A; // 804.8
    test_data[3][19] = 16'h625F; // 815.3
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[3][0] = 16'h6F36; // 7384.000000
    expected_flatten[3][1] = 16'h6856; // 2220.000000
    expected_flatten[3][2] = 16'h0000; // 0.000000
    expected_flatten[3][3] = 16'h6858; // 2224.000000
    expected_flatten[3][4] = 16'h6280; // 832.000000
    expected_flatten[3][5] = 16'h69C0; // 2944.000000
    expected_flatten[3][6] = 16'h5829; // 133.125000
    expected_flatten[3][7] = 16'h5C84; // 289.000000
    expected_flatten[3][8] = 16'h6C2D; // 4276.000000
    expected_flatten[3][9] = 16'h0000; // 0.000000
    expected_flatten[3][10] = 16'h5998; // 179.000000
    expected_flatten[3][11] = 16'h6252; // 809.000000
    expected_flatten[3][12] = 16'h64D1; // 1233.000000
    expected_flatten[3][13] = 16'h5515; // 81.312500
    expected_flatten[3][14] = 16'h0000; // 0.000000
    expected_flatten[3][15] = 16'h6F36; // 7384.000000
    expected_flatten[3][16] = 16'h6856; // 2220.000000
    expected_flatten[3][17] = 16'h0000; // 0.000000
    expected_flatten[3][18] = 16'h6858; // 2224.000000
    expected_flatten[3][19] = 16'h6280; // 832.000000
    expected_flatten[3][20] = 16'h69C0; // 2944.000000
    expected_flatten[3][21] = 16'h5829; // 133.125000
    expected_flatten[3][22] = 16'h5C84; // 289.000000
    expected_flatten[3][23] = 16'h6C2D; // 4276.000000

    // Test 4: Normal_SR_5
    expected_results[4] = 1'b1;
    test_data[4][0] = 16'h6264; // 818.2
    test_data[4][1] = 16'h6254; // 809.8
    test_data[4][2] = 16'h623B; // 797.7
    test_data[4][3] = 16'h6225; // 786.5
    test_data[4][4] = 16'h6219; // 780.4
    test_data[4][5] = 16'h621C; // 781.8
    test_data[4][6] = 16'h622C; // 790.2
    test_data[4][7] = 16'h6245; // 802.3
    test_data[4][8] = 16'h625B; // 813.5
    test_data[4][9] = 16'h6267; // 819.6
    test_data[4][10] = 16'h6264; // 818.2
    test_data[4][11] = 16'h6254; // 809.8
    test_data[4][12] = 16'h623B; // 797.7
    test_data[4][13] = 16'h6225; // 786.5
    test_data[4][14] = 16'h6219; // 780.4
    test_data[4][15] = 16'h621C; // 781.8
    test_data[4][16] = 16'h622C; // 790.2
    test_data[4][17] = 16'h6245; // 802.3
    test_data[4][18] = 16'h625B; // 813.5
    test_data[4][19] = 16'h6267; // 819.6
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[4][0] = 16'h6F36; // 7384.000000
    expected_flatten[4][1] = 16'h685C; // 2232.000000
    expected_flatten[4][2] = 16'h0000; // 0.000000
    expected_flatten[4][3] = 16'h682F; // 2142.000000
    expected_flatten[4][4] = 16'h625D; // 814.500000
    expected_flatten[4][5] = 16'h69CE; // 2972.000000
    expected_flatten[4][6] = 16'h5872; // 142.250000
    expected_flatten[4][7] = 16'h5CAA; // 298.500000
    expected_flatten[4][8] = 16'h6C33; // 4300.000000
    expected_flatten[4][9] = 16'h0000; // 0.000000
    expected_flatten[4][10] = 16'h59B2; // 182.250000
    expected_flatten[4][11] = 16'h6236; // 795.000000
    expected_flatten[4][12] = 16'h64DB; // 1243.000000
    expected_flatten[4][13] = 16'h5575; // 87.312500
    expected_flatten[4][14] = 16'h0000; // 0.000000
    expected_flatten[4][15] = 16'h6F36; // 7384.000000
    expected_flatten[4][16] = 16'h685C; // 2232.000000
    expected_flatten[4][17] = 16'h0000; // 0.000000
    expected_flatten[4][18] = 16'h682F; // 2142.000000
    expected_flatten[4][19] = 16'h625D; // 814.500000
    expected_flatten[4][20] = 16'h69CE; // 2972.000000
    expected_flatten[4][21] = 16'h5872; // 142.250000
    expected_flatten[4][22] = 16'h5CAA; // 298.500000
    expected_flatten[4][23] = 16'h6C33; // 4300.000000

    // Test 5: Tachycardia_1
    expected_results[5] = 1'b1;
    test_data[5][0] = 16'h5FD0; // 500.0
    test_data[5][1] = 16'h5FEC; // 507.1
    test_data[5][2] = 16'h5FF8; // 510.0
    test_data[5][3] = 16'h5FEC; // 507.1
    test_data[5][4] = 16'h5FD0; // 500.0
    test_data[5][5] = 16'h5FB4; // 492.9
    test_data[5][6] = 16'h5FA8; // 490.0
    test_data[5][7] = 16'h5FB4; // 492.9
    test_data[5][8] = 16'h5FD0; // 500.0
    test_data[5][9] = 16'h5FEC; // 507.1
    test_data[5][10] = 16'h5FF8; // 510.0
    test_data[5][11] = 16'h5FEC; // 507.1
    test_data[5][12] = 16'h5FD0; // 500.0
    test_data[5][13] = 16'h5FB4; // 492.9
    test_data[5][14] = 16'h5FA8; // 490.0
    test_data[5][15] = 16'h5FB4; // 492.9
    test_data[5][16] = 16'h5FD0; // 500.0
    test_data[5][17] = 16'h5FEC; // 507.1
    test_data[5][18] = 16'h5FF8; // 510.0
    test_data[5][19] = 16'h5FEC; // 507.1
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[5][0] = 16'h6C7E; // 4600.000000
    expected_flatten[5][1] = 16'h6561; // 1377.000000
    expected_flatten[5][2] = 16'h0000; // 0.000000
    expected_flatten[5][3] = 16'h6570; // 1392.000000
    expected_flatten[5][4] = 16'h6012; // 521.000000
    expected_flatten[5][5] = 16'h6746; // 1862.000000
    expected_flatten[5][6] = 16'h551C; // 81.750000
    expected_flatten[5][7] = 16'h591B; // 163.375000
    expected_flatten[5][8] = 16'h693B; // 2678.000000
    expected_flatten[5][9] = 16'h0000; // 0.000000
    expected_flatten[5][10] = 16'h56BB; // 107.687500
    expected_flatten[5][11] = 16'h5FF8; // 510.000000
    expected_flatten[5][12] = 16'h61FC; // 766.000000
    expected_flatten[5][13] = 16'h53FD; // 63.906250
    expected_flatten[5][14] = 16'h0000; // 0.000000
    expected_flatten[5][15] = 16'h6C7E; // 4600.000000
    expected_flatten[5][16] = 16'h6561; // 1377.000000
    expected_flatten[5][17] = 16'h0000; // 0.000000
    expected_flatten[5][18] = 16'h6570; // 1392.000000
    expected_flatten[5][19] = 16'h6012; // 521.000000
    expected_flatten[5][20] = 16'h6746; // 1862.000000
    expected_flatten[5][21] = 16'h551C; // 81.750000
    expected_flatten[5][22] = 16'h591B; // 163.375000
    expected_flatten[5][23] = 16'h693B; // 2678.000000

    // Test 6: Tachycardia_2
    expected_results[6] = 1'b1;
    test_data[6][0] = 16'h6024; // 530.0
    test_data[6][1] = 16'h6032; // 537.1
    test_data[6][2] = 16'h6038; // 540.0
    test_data[6][3] = 16'h6032; // 537.1
    test_data[6][4] = 16'h6024; // 530.0
    test_data[6][5] = 16'h6016; // 522.9
    test_data[6][6] = 16'h6010; // 520.0
    test_data[6][7] = 16'h6016; // 522.9
    test_data[6][8] = 16'h6024; // 530.0
    test_data[6][9] = 16'h6032; // 537.1
    test_data[6][10] = 16'h6038; // 540.0
    test_data[6][11] = 16'h6032; // 537.1
    test_data[6][12] = 16'h6024; // 530.0
    test_data[6][13] = 16'h6016; // 522.9
    test_data[6][14] = 16'h6010; // 520.0
    test_data[6][15] = 16'h6016; // 522.9
    test_data[6][16] = 16'h6024; // 530.0
    test_data[6][17] = 16'h6032; // 537.1
    test_data[6][18] = 16'h6038; // 540.0
    test_data[6][19] = 16'h6032; // 537.1
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[6][0] = 16'h6CC4; // 4880.000000
    expected_flatten[6][1] = 16'h65B5; // 1461.000000
    expected_flatten[6][2] = 16'h0000; // 0.000000
    expected_flatten[6][3] = 16'h65BE; // 1470.000000
    expected_flatten[6][4] = 16'h6047; // 547.500000
    expected_flatten[6][5] = 16'h67B0; // 1968.000000
    expected_flatten[6][6] = 16'h55AC; // 90.750000
    expected_flatten[6][7] = 16'h5984; // 176.500000
    expected_flatten[6][8] = 16'h6988; // 2832.000000
    expected_flatten[6][9] = 16'h0000; // 0.000000
    expected_flatten[6][10] = 16'h570B; // 112.687500
    expected_flatten[6][11] = 16'h603B; // 541.500000
    expected_flatten[6][12] = 16'h6255; // 810.500000
    expected_flatten[6][13] = 16'h5411; // 65.062500
    expected_flatten[6][14] = 16'h0000; // 0.000000
    expected_flatten[6][15] = 16'h6CC4; // 4880.000000
    expected_flatten[6][16] = 16'h65B5; // 1461.000000
    expected_flatten[6][17] = 16'h0000; // 0.000000
    expected_flatten[6][18] = 16'h65BE; // 1470.000000
    expected_flatten[6][19] = 16'h6047; // 547.500000
    expected_flatten[6][20] = 16'h67B0; // 1968.000000
    expected_flatten[6][21] = 16'h55AC; // 90.750000
    expected_flatten[6][22] = 16'h5984; // 176.500000
    expected_flatten[6][23] = 16'h6988; // 2832.000000

    // Test 7: Tachycardia_3
    expected_results[7] = 1'b1;
    test_data[7][0] = 16'h6060; // 560.0
    test_data[7][1] = 16'h606E; // 567.1
    test_data[7][2] = 16'h6074; // 570.0
    test_data[7][3] = 16'h606E; // 567.1
    test_data[7][4] = 16'h6060; // 560.0
    test_data[7][5] = 16'h6052; // 552.9
    test_data[7][6] = 16'h604C; // 550.0
    test_data[7][7] = 16'h6052; // 552.9
    test_data[7][8] = 16'h6060; // 560.0
    test_data[7][9] = 16'h606E; // 567.1
    test_data[7][10] = 16'h6074; // 570.0
    test_data[7][11] = 16'h606E; // 567.1
    test_data[7][12] = 16'h6060; // 560.0
    test_data[7][13] = 16'h6052; // 552.9
    test_data[7][14] = 16'h604C; // 550.0
    test_data[7][15] = 16'h6052; // 552.9
    test_data[7][16] = 16'h6060; // 560.0
    test_data[7][17] = 16'h606E; // 567.1
    test_data[7][18] = 16'h6074; // 570.0
    test_data[7][19] = 16'h606E; // 567.1
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[7][0] = 16'h6D09; // 5156.000000
    expected_flatten[7][1] = 16'h6606; // 1542.000000
    expected_flatten[7][2] = 16'h0000; // 0.000000
    expected_flatten[7][3] = 16'h6614; // 1556.000000
    expected_flatten[7][4] = 16'h608A; // 581.000000
    expected_flatten[7][5] = 16'h6810; // 2080.000000
    expected_flatten[7][6] = 16'h560C; // 96.750000
    expected_flatten[7][7] = 16'h59E3; // 188.375000
    expected_flatten[7][8] = 16'h69D8; // 2992.000000
    expected_flatten[7][9] = 16'h0000; // 0.000000
    expected_flatten[7][10] = 16'h575B; // 117.687500
    expected_flatten[7][11] = 16'h6074; // 570.000000
    expected_flatten[7][12] = 16'h62B0; // 856.000000
    expected_flatten[7][13] = 16'h544F; // 68.937500
    expected_flatten[7][14] = 16'h0000; // 0.000000
    expected_flatten[7][15] = 16'h6D09; // 5156.000000
    expected_flatten[7][16] = 16'h6606; // 1542.000000
    expected_flatten[7][17] = 16'h0000; // 0.000000
    expected_flatten[7][18] = 16'h6614; // 1556.000000
    expected_flatten[7][19] = 16'h608A; // 581.000000
    expected_flatten[7][20] = 16'h6810; // 2080.000000
    expected_flatten[7][21] = 16'h560C; // 96.750000
    expected_flatten[7][22] = 16'h59E3; // 188.375000
    expected_flatten[7][23] = 16'h69D8; // 2992.000000

    // Test 8: Bradycardia_1
    expected_results[8] = 1'b1;
    test_data[8][0] = 16'h63D0; // 1000.0
    test_data[8][1] = 16'h63DF; // 1007.5
    test_data[8][2] = 16'h63EA; // 1013.0
    test_data[8][3] = 16'h63EE; // 1015.0
    test_data[8][4] = 16'h63EA; // 1013.0
    test_data[8][5] = 16'h63DF; // 1007.5
    test_data[8][6] = 16'h63D0; // 1000.0
    test_data[8][7] = 16'h63C1; // 992.5
    test_data[8][8] = 16'h63B6; // 987.0
    test_data[8][9] = 16'h63B2; // 985.0
    test_data[8][10] = 16'h63B6; // 987.0
    test_data[8][11] = 16'h63C1; // 992.5
    test_data[8][12] = 16'h63D0; // 1000.0
    test_data[8][13] = 16'h63DF; // 1007.5
    test_data[8][14] = 16'h63EA; // 1013.0
    test_data[8][15] = 16'h63EE; // 1015.0
    test_data[8][16] = 16'h63EA; // 1013.0
    test_data[8][17] = 16'h63DF; // 1007.5
    test_data[8][18] = 16'h63D0; // 1000.0
    test_data[8][19] = 16'h63C1; // 992.5
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[8][0] = 16'h708C; // 9312.000000
    expected_flatten[8][1] = 16'h6979; // 2802.000000
    expected_flatten[8][2] = 16'h5030; // 33.500000
    expected_flatten[8][3] = 16'h6978; // 2800.000000
    expected_flatten[8][4] = 16'h641C; // 1052.000000
    expected_flatten[8][5] = 16'h6B34; // 3688.000000
    expected_flatten[8][6] = 16'h5CA0; // 296.000000
    expected_flatten[8][7] = 16'h5F98; // 486.000000
    expected_flatten[8][8] = 16'h6D26; // 5272.000000
    expected_flatten[8][9] = 16'h0000; // 0.000000
    expected_flatten[8][10] = 16'h5989; // 177.125000
    expected_flatten[8][11] = 16'h63EB; // 1013.500000
    expected_flatten[8][12] = 16'h65BF; // 1471.000000
    expected_flatten[8][13] = 16'h582F; // 133.875000
    expected_flatten[8][14] = 16'h524C; // 50.375000
    expected_flatten[8][15] = 16'h708C; // 9312.000000
    expected_flatten[8][16] = 16'h6979; // 2802.000000
    expected_flatten[8][17] = 16'h5030; // 33.500000
    expected_flatten[8][18] = 16'h6978; // 2800.000000
    expected_flatten[8][19] = 16'h641C; // 1052.000000
    expected_flatten[8][20] = 16'h6B34; // 3688.000000
    expected_flatten[8][21] = 16'h5CA0; // 296.000000
    expected_flatten[8][22] = 16'h5F98; // 486.000000
    expected_flatten[8][23] = 16'h6D26; // 5272.000000

    // Test 9: Bradycardia_2
    expected_results[9] = 1'b1;
    test_data[9][0] = 16'h641A; // 1050.0
    test_data[9][1] = 16'h6422; // 1057.5
    test_data[9][2] = 16'h6427; // 1063.0
    test_data[9][3] = 16'h6429; // 1065.0
    test_data[9][4] = 16'h6427; // 1063.0
    test_data[9][5] = 16'h6422; // 1057.5
    test_data[9][6] = 16'h641A; // 1050.0
    test_data[9][7] = 16'h6412; // 1042.5
    test_data[9][8] = 16'h640D; // 1037.0
    test_data[9][9] = 16'h640B; // 1035.0
    test_data[9][10] = 16'h640D; // 1037.0
    test_data[9][11] = 16'h6412; // 1042.5
    test_data[9][12] = 16'h641A; // 1050.0
    test_data[9][13] = 16'h6422; // 1057.5
    test_data[9][14] = 16'h6427; // 1063.0
    test_data[9][15] = 16'h6429; // 1065.0
    test_data[9][16] = 16'h6427; // 1063.0
    test_data[9][17] = 16'h6422; // 1057.5
    test_data[9][18] = 16'h641A; // 1050.0
    test_data[9][19] = 16'h6412; // 1042.5
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[9][0] = 16'h70C6; // 9776.000000
    expected_flatten[9][1] = 16'h69BD; // 2938.000000
    expected_flatten[9][2] = 16'h5040; // 34.000000
    expected_flatten[9][3] = 16'h69BE; // 2940.000000
    expected_flatten[9][4] = 16'h6450; // 1104.000000
    expected_flatten[9][5] = 16'h6B90; // 3872.000000
    expected_flatten[9][6] = 16'h5CD8; // 310.000000
    expected_flatten[9][7] = 16'h5FF4; // 509.000000
    expected_flatten[9][8] = 16'h6D68; // 5536.000000
    expected_flatten[9][9] = 16'h0000; // 0.000000
    expected_flatten[9][10] = 16'h59CE; // 185.750000
    expected_flatten[9][11] = 16'h6426; // 1062.000000
    expected_flatten[9][12] = 16'h6609; // 1545.000000
    expected_flatten[9][13] = 16'h5865; // 140.625000
    expected_flatten[9][14] = 16'h5290; // 52.500000
    expected_flatten[9][15] = 16'h70C6; // 9776.000000
    expected_flatten[9][16] = 16'h69BD; // 2938.000000
    expected_flatten[9][17] = 16'h5040; // 34.000000
    expected_flatten[9][18] = 16'h69BE; // 2940.000000
    expected_flatten[9][19] = 16'h6450; // 1104.000000
    expected_flatten[9][20] = 16'h6B90; // 3872.000000
    expected_flatten[9][21] = 16'h5CD8; // 310.000000
    expected_flatten[9][22] = 16'h5FF4; // 509.000000
    expected_flatten[9][23] = 16'h6D68; // 5536.000000

    // Test 10: Bradycardia_3
    expected_results[10] = 1'b1;
    test_data[10][0] = 16'h644C; // 1100.0
    test_data[10][1] = 16'h6454; // 1107.5
    test_data[10][2] = 16'h6459; // 1113.0
    test_data[10][3] = 16'h645B; // 1115.0
    test_data[10][4] = 16'h6459; // 1113.0
    test_data[10][5] = 16'h6454; // 1107.5
    test_data[10][6] = 16'h644C; // 1100.0
    test_data[10][7] = 16'h6444; // 1092.5
    test_data[10][8] = 16'h643F; // 1087.0
    test_data[10][9] = 16'h643D; // 1085.0
    test_data[10][10] = 16'h643F; // 1087.0
    test_data[10][11] = 16'h6444; // 1092.5
    test_data[10][12] = 16'h644C; // 1100.0
    test_data[10][13] = 16'h6454; // 1107.5
    test_data[10][14] = 16'h6459; // 1113.0
    test_data[10][15] = 16'h645B; // 1115.0
    test_data[10][16] = 16'h6459; // 1113.0
    test_data[10][17] = 16'h6454; // 1107.5
    test_data[10][18] = 16'h644C; // 1100.0
    test_data[10][19] = 16'h6444; // 1092.5
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[10][0] = 16'h7100; // 10240.000000
    expected_flatten[10][1] = 16'h6A02; // 3076.000000
    expected_flatten[10][2] = 16'h5020; // 33.000000
    expected_flatten[10][3] = 16'h6A01; // 3074.000000
    expected_flatten[10][4] = 16'h6480; // 1152.000000
    expected_flatten[10][5] = 16'h6BE9; // 4050.000000
    expected_flatten[10][6] = 16'h5D12; // 324.500000
    expected_flatten[10][7] = 16'h6029; // 532.500000
    expected_flatten[10][8] = 16'h6DAA; // 5800.000000
    expected_flatten[10][9] = 16'h0000; // 0.000000
    expected_flatten[10][10] = 16'h5A2A; // 197.250000
    expected_flatten[10][11] = 16'h6458; // 1112.000000
    expected_flatten[10][12] = 16'h6654; // 1620.000000
    expected_flatten[10][13] = 16'h589B; // 147.375000
    expected_flatten[10][14] = 16'h52D0; // 54.500000
    expected_flatten[10][15] = 16'h7100; // 10240.000000
    expected_flatten[10][16] = 16'h6A02; // 3076.000000
    expected_flatten[10][17] = 16'h5020; // 33.000000
    expected_flatten[10][18] = 16'h6A01; // 3074.000000
    expected_flatten[10][19] = 16'h6480; // 1152.000000
    expected_flatten[10][20] = 16'h6BE9; // 4050.000000
    expected_flatten[10][21] = 16'h5D12; // 324.500000
    expected_flatten[10][22] = 16'h6029; // 532.500000
    expected_flatten[10][23] = 16'h6DAA; // 5800.000000

    // Test 11: AFib_1
    expected_results[11] = 1'b1;
    test_data[11][0] = 16'h6114; // 650.0
    test_data[11][1] = 16'h6330; // 920.0
    test_data[11][2] = 16'h6088; // 580.0
    test_data[11][3] = 16'h641A; // 1050.0
    test_data[11][4] = 16'h61A0; // 720.0
    test_data[11][5] = 16'h62F4; // 890.0
    test_data[11][6] = 16'h60C4; // 610.0
    test_data[11][7] = 16'h644C; // 1100.0
    test_data[11][8] = 16'h604C; // 550.0
    test_data[11][9] = 16'h63A8; // 980.0
    test_data[11][10] = 16'h6218; // 780.0
    test_data[11][11] = 16'h6100; // 640.0
    test_data[11][12] = 16'h63F8; // 1020.0
    test_data[11][13] = 16'h609C; // 590.0
    test_data[11][14] = 16'h636C; // 950.0
    test_data[11][15] = 16'h6178; // 700.0
    test_data[11][16] = 16'h647E; // 1150.0
    test_data[11][17] = 16'h6024; // 530.0
    test_data[11][18] = 16'h62E0; // 880.0
    test_data[11][19] = 16'h613C; // 670.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[11][0] = 16'h6F92; // 7752.000000
    expected_flatten[11][1] = 16'h6A9A; // 3380.000000
    expected_flatten[11][2] = 16'h65E6; // 1510.000000
    expected_flatten[11][3] = 16'h5F0F; // 451.750000
    expected_flatten[11][4] = 16'h65E4; // 1508.000000
    expected_flatten[11][5] = 16'h6D32; // 5320.000000
    expected_flatten[11][6] = 16'h5CCD; // 307.250000
    expected_flatten[11][7] = 16'h5E67; // 409.750000
    expected_flatten[11][8] = 16'h6CFE; // 5112.000000
    expected_flatten[11][9] = 16'h0000; // 0.000000
    expected_flatten[11][10] = 16'h5C7D; // 287.250000
    expected_flatten[11][11] = 16'h6552; // 1362.000000
    expected_flatten[11][12] = 16'h697A; // 2804.000000
    expected_flatten[11][13] = 16'h6283; // 833.500000
    expected_flatten[11][14] = 16'h5B94; // 242.500000
    expected_flatten[11][15] = 16'h6F92; // 7752.000000
    expected_flatten[11][16] = 16'h6A9A; // 3380.000000
    expected_flatten[11][17] = 16'h65E6; // 1510.000000
    expected_flatten[11][18] = 16'h5F0F; // 451.750000
    expected_flatten[11][19] = 16'h65E4; // 1508.000000
    expected_flatten[11][20] = 16'h6D32; // 5320.000000
    expected_flatten[11][21] = 16'h5CCD; // 307.250000
    expected_flatten[11][22] = 16'h5E67; // 409.750000
    expected_flatten[11][23] = 16'h6CFE; // 5112.000000

    // Test 12: AFib_2
    expected_results[12] = 1'b1;
    test_data[12][0] = 16'h6178; // 700.0
    test_data[12][1] = 16'h62A4; // 850.0
    test_data[12][2] = 16'h60D8; // 620.0
    test_data[12][3] = 16'h63A8; // 980.0
    test_data[12][4] = 16'h61DC; // 750.0
    test_data[12][5] = 16'h6308; // 900.0
    test_data[12][6] = 16'h6088; // 580.0
    test_data[12][7] = 16'h6438; // 1080.0
    test_data[12][8] = 16'h60B0; // 600.0
    test_data[12][9] = 16'h636C; // 950.0
    test_data[12][10] = 16'h61A0; // 720.0
    test_data[12][11] = 16'h6150; // 680.0
    test_data[12][12] = 16'h63D0; // 1000.0
    test_data[12][13] = 16'h604C; // 550.0
    test_data[12][14] = 16'h6330; // 920.0
    test_data[12][15] = 16'h61F0; // 760.0
    test_data[12][16] = 16'h641A; // 1050.0
    test_data[12][17] = 16'h609C; // 590.0
    test_data[12][18] = 16'h62CC; // 870.0
    test_data[12][19] = 16'h6100; // 640.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[12][0] = 16'h6EF0; // 7104.000000
    expected_flatten[12][1] = 16'h6A1A; // 3124.000000
    expected_flatten[12][2] = 16'h668A; // 1674.000000
    expected_flatten[12][3] = 16'h6002; // 513.000000
    expected_flatten[12][4] = 16'h6680; // 1664.000000
    expected_flatten[12][5] = 16'h6D63; // 5516.000000
    expected_flatten[12][6] = 16'h0000; // 0.000000
    expected_flatten[12][7] = 16'h59BD; // 183.625000
    expected_flatten[12][8] = 16'h6CE6; // 5016.000000
    expected_flatten[12][9] = 16'h0000; // 0.000000
    expected_flatten[12][10] = 16'h5E35; // 397.250000
    expected_flatten[12][11] = 16'h66AB; // 1707.000000
    expected_flatten[12][12] = 16'h6921; // 2626.000000
    expected_flatten[12][13] = 16'h6262; // 817.000000
    expected_flatten[12][14] = 16'h5CAC; // 299.000000
    expected_flatten[12][15] = 16'h6EF0; // 7104.000000
    expected_flatten[12][16] = 16'h6A1A; // 3124.000000
    expected_flatten[12][17] = 16'h668A; // 1674.000000
    expected_flatten[12][18] = 16'h6002; // 513.000000
    expected_flatten[12][19] = 16'h6680; // 1664.000000
    expected_flatten[12][20] = 16'h6D63; // 5516.000000
    expected_flatten[12][21] = 16'h0000; // 0.000000
    expected_flatten[12][22] = 16'h59BD; // 183.625000
    expected_flatten[12][23] = 16'h6CE6; // 5016.000000

    // Test 13: AFib_3
    expected_results[13] = 1'b1;
    test_data[13][0] = 16'h6150; // 680.0
    test_data[13][1] = 16'h636C; // 950.0
    test_data[13][2] = 16'h60B0; // 600.0
    test_data[13][3] = 16'h63F8; // 1020.0
    test_data[13][4] = 16'h618C; // 710.0
    test_data[13][5] = 16'h62E0; // 880.0
    test_data[13][6] = 16'h60EC; // 630.0
    test_data[13][7] = 16'h6460; // 1120.0
    test_data[13][8] = 16'h6074; // 570.0
    test_data[13][9] = 16'h63BC; // 990.0
    test_data[13][10] = 16'h61C8; // 740.0
    test_data[13][11] = 16'h6114; // 650.0
    test_data[13][12] = 16'h6406; // 1030.0
    test_data[13][13] = 16'h6088; // 580.0
    test_data[13][14] = 16'h6380; // 960.0
    test_data[13][15] = 16'h61B4; // 730.0
    test_data[13][16] = 16'h6438; // 1080.0
    test_data[13][17] = 16'h6060; // 560.0
    test_data[13][18] = 16'h62F4; // 890.0
    test_data[13][19] = 16'h6128; // 660.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[13][0] = 16'h6F79; // 7652.000000
    expected_flatten[13][1] = 16'h6A79; // 3314.000000
    expected_flatten[13][2] = 16'h66A1; // 1697.000000
    expected_flatten[13][3] = 16'h5FFA; // 510.500000
    expected_flatten[13][4] = 16'h667D; // 1661.000000
    expected_flatten[13][5] = 16'h6D5E; // 5496.000000
    expected_flatten[13][6] = 16'h5CC7; // 305.750000
    expected_flatten[13][7] = 16'h5E5F; // 407.750000
    expected_flatten[13][8] = 16'h6D08; // 5152.000000
    expected_flatten[13][9] = 16'h0000; // 0.000000
    expected_flatten[13][10] = 16'h5E27; // 393.750000
    expected_flatten[13][11] = 16'h6600; // 1536.000000
    expected_flatten[13][12] = 16'h698E; // 2844.000000
    expected_flatten[13][13] = 16'h634C; // 934.000000
    expected_flatten[13][14] = 16'h5C67; // 281.750000
    expected_flatten[13][15] = 16'h6F79; // 7652.000000
    expected_flatten[13][16] = 16'h6A79; // 3314.000000
    expected_flatten[13][17] = 16'h66A1; // 1697.000000
    expected_flatten[13][18] = 16'h5FFA; // 510.500000
    expected_flatten[13][19] = 16'h667D; // 1661.000000
    expected_flatten[13][20] = 16'h6D5E; // 5496.000000
    expected_flatten[13][21] = 16'h5CC7; // 305.750000
    expected_flatten[13][22] = 16'h5E5F; // 407.750000
    expected_flatten[13][23] = 16'h6D08; // 5152.000000

    // Test 14: AFib_4
    expected_results[14] = 1'b1;
    test_data[14][0] = 16'h61A0; // 720.0
    test_data[14][1] = 16'h62F4; // 890.0
    test_data[14][2] = 16'h6100; // 640.0
    test_data[14][3] = 16'h63D0; // 1000.0
    test_data[14][4] = 16'h61B4; // 730.0
    test_data[14][5] = 16'h631C; // 910.0
    test_data[14][6] = 16'h609C; // 590.0
    test_data[14][7] = 16'h6424; // 1060.0
    test_data[14][8] = 16'h60D8; // 620.0
    test_data[14][9] = 16'h6394; // 970.0
    test_data[14][10] = 16'h61F0; // 760.0
    test_data[14][11] = 16'h613C; // 670.0
    test_data[14][12] = 16'h63E4; // 1010.0
    test_data[14][13] = 16'h60B0; // 600.0
    test_data[14][14] = 16'h6344; // 930.0
    test_data[14][15] = 16'h618C; // 710.0
    test_data[14][16] = 16'h6442; // 1090.0
    test_data[14][17] = 16'h6074; // 570.0
    test_data[14][18] = 16'h6308; // 900.0
    test_data[14][19] = 16'h6150; // 680.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[14][0] = 16'h6F09; // 7204.000000
    expected_flatten[14][1] = 16'h6A5A; // 3252.000000
    expected_flatten[14][2] = 16'h66AC; // 1708.000000
    expected_flatten[14][3] = 16'h6078; // 572.000000
    expected_flatten[14][4] = 16'h662D; // 1581.000000
    expected_flatten[14][5] = 16'h6D2C; // 5296.000000
    expected_flatten[14][6] = 16'h50C8; // 38.250000
    expected_flatten[14][7] = 16'h5A31; // 198.125000
    expected_flatten[14][8] = 16'h6D00; // 5120.000000
    expected_flatten[14][9] = 16'h0000; // 0.000000
    expected_flatten[14][10] = 16'h5EC7; // 433.750000
    expected_flatten[14][11] = 16'h65D9; // 1497.000000
    expected_flatten[14][12] = 16'h6950; // 2720.000000
    expected_flatten[14][13] = 16'h6295; // 842.500000
    expected_flatten[14][14] = 16'h5CC9; // 306.250000
    expected_flatten[14][15] = 16'h6F09; // 7204.000000
    expected_flatten[14][16] = 16'h6A5A; // 3252.000000
    expected_flatten[14][17] = 16'h66AC; // 1708.000000
    expected_flatten[14][18] = 16'h6078; // 572.000000
    expected_flatten[14][19] = 16'h662D; // 1581.000000
    expected_flatten[14][20] = 16'h6D2C; // 5296.000000
    expected_flatten[14][21] = 16'h50C8; // 38.250000
    expected_flatten[14][22] = 16'h5A31; // 198.125000
    expected_flatten[14][23] = 16'h6D00; // 5120.000000

    // Test 15: Mixed_1
    expected_results[15] = 1'b1;
    test_data[15][0] = 16'h62A4; // 850.0
    test_data[15][1] = 16'h6218; // 780.0
    test_data[15][2] = 16'h6330; // 920.0
    test_data[15][3] = 16'h6254; // 810.0
    test_data[15][4] = 16'h62E0; // 880.0
    test_data[15][5] = 16'h622C; // 790.0
    test_data[15][6] = 16'h6308; // 900.0
    test_data[15][7] = 16'h6268; // 820.0
    test_data[15][8] = 16'h62B8; // 860.0
    test_data[15][9] = 16'h627C; // 830.0
    test_data[15][10] = 16'h62CC; // 870.0
    test_data[15][11] = 16'h6240; // 800.0
    test_data[15][12] = 16'h631C; // 910.0
    test_data[15][13] = 16'h6290; // 840.0
    test_data[15][14] = 16'h62F4; // 890.0
    test_data[15][15] = 16'h6254; // 810.0
    test_data[15][16] = 16'h62E0; // 880.0
    test_data[15][17] = 16'h6268; // 820.0
    test_data[15][18] = 16'h6308; // 900.0
    test_data[15][19] = 16'h62A4; // 850.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[15][0] = 16'h6F9B; // 7788.000000
    expected_flatten[15][1] = 16'h6926; // 2636.000000
    expected_flatten[15][2] = 16'h600C; // 518.000000
    expected_flatten[15][3] = 16'h66C7; // 1735.000000
    expected_flatten[15][4] = 16'h62C3; // 865.500000
    expected_flatten[15][5] = 16'h6B7B; // 3830.000000
    expected_flatten[15][6] = 16'h58DC; // 155.500000
    expected_flatten[15][7] = 16'h5C53; // 276.750000
    expected_flatten[15][8] = 16'h6CDC; // 4976.000000
    expected_flatten[15][9] = 16'h0000; // 0.000000
    expected_flatten[15][10] = 16'h4B40; // 14.500000
    expected_flatten[15][11] = 16'h6182; // 705.000000
    expected_flatten[15][12] = 16'h65A8; // 1448.000000
    expected_flatten[15][13] = 16'h58A0; // 148.000000
    expected_flatten[15][14] = 16'h5268; // 51.250000
    expected_flatten[15][15] = 16'h6F9B; // 7788.000000
    expected_flatten[15][16] = 16'h6926; // 2636.000000
    expected_flatten[15][17] = 16'h600C; // 518.000000
    expected_flatten[15][18] = 16'h66C7; // 1735.000000
    expected_flatten[15][19] = 16'h62C3; // 865.500000
    expected_flatten[15][20] = 16'h6B7B; // 3830.000000
    expected_flatten[15][21] = 16'h58DC; // 155.500000
    expected_flatten[15][22] = 16'h5C53; // 276.750000
    expected_flatten[15][23] = 16'h6CDC; // 4976.000000

    // Test 16: Mixed_2
    expected_results[16] = 1'b1;
    test_data[16][0] = 16'h61DC; // 750.0
    test_data[16][1] = 16'h636C; // 950.0
    test_data[16][2] = 16'h6150; // 680.0
    test_data[16][3] = 16'h63F8; // 1020.0
    test_data[16][4] = 16'h622C; // 790.0
    test_data[16][5] = 16'h62E0; // 880.0
    test_data[16][6] = 16'h61A0; // 720.0
    test_data[16][7] = 16'h63A8; // 980.0
    test_data[16][8] = 16'h6254; // 810.0
    test_data[16][9] = 16'h6308; // 900.0
    test_data[16][10] = 16'h61F0; // 760.0
    test_data[16][11] = 16'h6358; // 940.0
    test_data[16][12] = 16'h6178; // 700.0
    test_data[16][13] = 16'h63D0; // 1000.0
    test_data[16][14] = 16'h6218; // 780.0
    test_data[16][15] = 16'h6330; // 920.0
    test_data[16][16] = 16'h61C8; // 740.0
    test_data[16][17] = 16'h6380; // 960.0
    test_data[16][18] = 16'h6240; // 800.0
    test_data[16][19] = 16'h62F4; // 890.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[16][0] = 16'h6F6D; // 7604.000000
    expected_flatten[16][1] = 16'h6951; // 2722.000000
    expected_flatten[16][2] = 16'h5A3C; // 199.500000
    expected_flatten[16][3] = 16'h6401; // 1025.000000
    expected_flatten[16][4] = 16'h61D2; // 745.000000
    expected_flatten[16][5] = 16'h6B7C; // 3832.000000
    expected_flatten[16][6] = 16'h0000; // 0.000000
    expected_flatten[16][7] = 16'h5D0B; // 322.750000
    expected_flatten[16][8] = 16'h6CEA; // 5032.000000
    expected_flatten[16][9] = 16'h0000; // 0.000000
    expected_flatten[16][10] = 16'h5D4C; // 339.000000
    expected_flatten[16][11] = 16'h5F1A; // 454.500000
    expected_flatten[16][12] = 16'h6881; // 2306.000000
    expected_flatten[16][13] = 16'h6042; // 545.000000
    expected_flatten[16][14] = 16'h55DC; // 93.750000
    expected_flatten[16][15] = 16'h6F6D; // 7604.000000
    expected_flatten[16][16] = 16'h6951; // 2722.000000
    expected_flatten[16][17] = 16'h5A3C; // 199.500000
    expected_flatten[16][18] = 16'h6401; // 1025.000000
    expected_flatten[16][19] = 16'h61D2; // 745.000000
    expected_flatten[16][20] = 16'h6B7C; // 3832.000000
    expected_flatten[16][21] = 16'h0000; // 0.000000
    expected_flatten[16][22] = 16'h5D0B; // 322.750000
    expected_flatten[16][23] = 16'h6CEA; // 5032.000000

    // Test 17: Mixed_3
    expected_results[17] = 1'b1;
    test_data[17][0] = 16'h60B0; // 600.0
    test_data[17][1] = 16'h62A4; // 850.0
    test_data[17][2] = 16'h644C; // 1100.0
    test_data[17][3] = 16'h6178; // 700.0
    test_data[17][4] = 16'h636C; // 950.0
    test_data[17][5] = 16'h6114; // 650.0
    test_data[17][6] = 16'h63D0; // 1000.0
    test_data[17][7] = 16'h61DC; // 750.0
    test_data[17][8] = 16'h6308; // 900.0
    test_data[17][9] = 16'h6240; // 800.0
    test_data[17][10] = 16'h641A; // 1050.0
    test_data[17][11] = 16'h6150; // 680.0
    test_data[17][12] = 16'h63A8; // 980.0
    test_data[17][13] = 16'h61A0; // 720.0
    test_data[17][14] = 16'h6330; // 920.0
    test_data[17][15] = 16'h6128; // 660.0
    test_data[17][16] = 16'h63E4; // 1010.0
    test_data[17][17] = 16'h6218; // 780.0
    test_data[17][18] = 16'h62E0; // 880.0
    test_data[17][19] = 16'h6178; // 700.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[17][0] = 16'h6EEC; // 7088.000000
    expected_flatten[17][1] = 16'h69AC; // 2904.000000
    expected_flatten[17][2] = 16'h5700; // 112.000000
    expected_flatten[17][3] = 16'h6065; // 562.500000
    expected_flatten[17][4] = 16'h63E0; // 1008.000000
    expected_flatten[17][5] = 16'h6C7E; // 4600.000000
    expected_flatten[17][6] = 16'h0000; // 0.000000
    expected_flatten[17][7] = 16'h5D71; // 348.250000
    expected_flatten[17][8] = 16'h6D2E; // 5304.000000
    expected_flatten[17][9] = 16'h5892; // 146.250000
    expected_flatten[17][10] = 16'h5BC9; // 249.125000
    expected_flatten[17][11] = 16'h62D6; // 875.000000
    expected_flatten[17][12] = 16'h6876; // 2284.000000
    expected_flatten[17][13] = 16'h6262; // 817.000000
    expected_flatten[17][14] = 16'h5158; // 42.750000
    expected_flatten[17][15] = 16'h6EEC; // 7088.000000
    expected_flatten[17][16] = 16'h69AC; // 2904.000000
    expected_flatten[17][17] = 16'h5700; // 112.000000
    expected_flatten[17][18] = 16'h6065; // 562.500000
    expected_flatten[17][19] = 16'h63E0; // 1008.000000
    expected_flatten[17][20] = 16'h6C7E; // 4600.000000
    expected_flatten[17][21] = 16'h0000; // 0.000000
    expected_flatten[17][22] = 16'h5D71; // 348.250000
    expected_flatten[17][23] = 16'h6D2E; // 5304.000000

    // Test 18: All_Min_300ms
    expected_results[18] = 1'b1;
    test_data[18][0] = 16'h5CB0; // 300.0
    test_data[18][1] = 16'h5CB0; // 300.0
    test_data[18][2] = 16'h5CB0; // 300.0
    test_data[18][3] = 16'h5CB0; // 300.0
    test_data[18][4] = 16'h5CB0; // 300.0
    test_data[18][5] = 16'h5CB0; // 300.0
    test_data[18][6] = 16'h5CB0; // 300.0
    test_data[18][7] = 16'h5CB0; // 300.0
    test_data[18][8] = 16'h5CB0; // 300.0
    test_data[18][9] = 16'h5CB0; // 300.0
    test_data[18][10] = 16'h5CB0; // 300.0
    test_data[18][11] = 16'h5CB0; // 300.0
    test_data[18][12] = 16'h5CB0; // 300.0
    test_data[18][13] = 16'h5CB0; // 300.0
    test_data[18][14] = 16'h5CB0; // 300.0
    test_data[18][15] = 16'h5CB0; // 300.0
    test_data[18][16] = 16'h5CB0; // 300.0
    test_data[18][17] = 16'h5CB0; // 300.0
    test_data[18][18] = 16'h5CB0; // 300.0
    test_data[18][19] = 16'h5CB0; // 300.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[18][0] = 16'h696C; // 2776.000000
    expected_flatten[18][1] = 16'h6273; // 825.500000
    expected_flatten[18][2] = 16'h0000; // 0.000000
    expected_flatten[18][3] = 16'h6272; // 825.000000
    expected_flatten[18][4] = 16'h5CB0; // 300.000000
    expected_flatten[18][5] = 16'h6440; // 1088.000000
    expected_flatten[18][6] = 16'h54B0; // 75.000000
    expected_flatten[18][7] = 16'h57D0; // 125.000000
    expected_flatten[18][8] = 16'h6628; // 1576.000000
    expected_flatten[18][9] = 16'h0000; // 0.000000
    expected_flatten[18][10] = 16'h5240; // 50.000000
    expected_flatten[18][11] = 16'h5CB0; // 300.000000
    expected_flatten[18][12] = 16'h5F08; // 450.000000
    expected_flatten[18][13] = 16'h4E40; // 25.000000
    expected_flatten[18][14] = 16'h0000; // 0.000000
    expected_flatten[18][15] = 16'h696C; // 2776.000000
    expected_flatten[18][16] = 16'h6273; // 825.500000
    expected_flatten[18][17] = 16'h0000; // 0.000000
    expected_flatten[18][18] = 16'h6272; // 825.000000
    expected_flatten[18][19] = 16'h5CB0; // 300.000000
    expected_flatten[18][20] = 16'h6440; // 1088.000000
    expected_flatten[18][21] = 16'h54B0; // 75.000000
    expected_flatten[18][22] = 16'h57D0; // 125.000000
    expected_flatten[18][23] = 16'h6628; // 1576.000000

    // Test 19: All_Max_2000ms
    expected_results[19] = 1'b1;
    test_data[19][0] = 16'h67D0; // 2000.0
    test_data[19][1] = 16'h67D0; // 2000.0
    test_data[19][2] = 16'h67D0; // 2000.0
    test_data[19][3] = 16'h67D0; // 2000.0
    test_data[19][4] = 16'h67D0; // 2000.0
    test_data[19][5] = 16'h67D0; // 2000.0
    test_data[19][6] = 16'h67D0; // 2000.0
    test_data[19][7] = 16'h67D0; // 2000.0
    test_data[19][8] = 16'h67D0; // 2000.0
    test_data[19][9] = 16'h67D0; // 2000.0
    test_data[19][10] = 16'h67D0; // 2000.0
    test_data[19][11] = 16'h67D0; // 2000.0
    test_data[19][12] = 16'h67D0; // 2000.0
    test_data[19][13] = 16'h67D0; // 2000.0
    test_data[19][14] = 16'h67D0; // 2000.0
    test_data[19][15] = 16'h67D0; // 2000.0
    test_data[19][16] = 16'h67D0; // 2000.0
    test_data[19][17] = 16'h67D0; // 2000.0
    test_data[19][18] = 16'h67D0; // 2000.0
    test_data[19][19] = 16'h67D0; // 2000.0
    // 期望展平层输出 (Flatten 24 values)
    expected_flatten[19][0] = 16'h7484; // 18496.000000
    expected_flatten[19][1] = 16'h6D5F; // 5500.000000
    expected_flatten[19][2] = 16'h0000; // 0.000000
    expected_flatten[19][3] = 16'h6D5F; // 5500.000000
    expected_flatten[19][4] = 16'h67D0; // 2000.000000
    expected_flatten[19][5] = 16'h6F14; // 7248.000000
    expected_flatten[19][6] = 16'h5FD0; // 500.000000
    expected_flatten[19][7] = 16'h6283; // 833.500000
    expected_flatten[19][8] = 16'h7120; // 10496.000000
    expected_flatten[19][9] = 16'h0000; // 0.000000
    expected_flatten[19][10] = 16'h5D35; // 333.250000
    expected_flatten[19][11] = 16'h67D0; // 2000.000000
    expected_flatten[19][12] = 16'h69DC; // 3000.000000
    expected_flatten[19][13] = 16'h5935; // 166.625000
    expected_flatten[19][14] = 16'h0000; // 0.000000
    expected_flatten[19][15] = 16'h7484; // 18496.000000
    expected_flatten[19][16] = 16'h6D5F; // 5500.000000
    expected_flatten[19][17] = 16'h0000; // 0.000000
    expected_flatten[19][18] = 16'h6D5F; // 5500.000000
    expected_flatten[19][19] = 16'h67D0; // 2000.000000
    expected_flatten[19][20] = 16'h6F14; // 7248.000000
    expected_flatten[19][21] = 16'h5FD0; // 500.000000
    expected_flatten[19][22] = 16'h6283; // 833.500000
    expected_flatten[19][23] = 16'h7120; // 10496.000000


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
