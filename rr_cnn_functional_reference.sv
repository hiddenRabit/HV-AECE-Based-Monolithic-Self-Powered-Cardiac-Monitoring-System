//=============================================================================
// RR间期二分类网络 - FP16版本
// 使用IEEE-754半精度浮点数
//=============================================================================

module rr_classifier_fp16 #(
    parameter DATA_WIDTH = 16,      // FP16
    parameter SEQ_LEN = 20,
    parameter POOL_LEN = 3
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,
    input  wire [DATA_WIDTH-1:0]   rr_data [0:SEQ_LEN-1],
    output reg                     done,
    output reg                     prediction,
    output reg  [DATA_WIDTH-1:0]   logits_out [0:1]
);

//=============================================================================
// FP16权重存储
//=============================================================================
reg [DATA_WIDTH-1:0] conv1_weight [0:2][0:6];
reg [DATA_WIDTH-1:0] conv2_weight [0:7][0:2][0:4];
reg [DATA_WIDTH-1:0] fc_weight [0:1][0:23];
reg [DATA_WIDTH-1:0] fc_bias [0:1];

// 权重初始化（从Python生成）
initial begin
    // Conv1权重
    conv1_weight[0][0] = 16'h3C00; // 1.0
    conv1_weight[0][1] = 16'hBC00; // -1.0
    conv1_weight[0][2] = 16'h4000; // 2.0
    conv1_weight[0][3] = 16'h4200; // 3.0
    conv1_weight[0][4] = 16'h4000; // 2.0
    conv1_weight[0][5] = 16'hBC00; // -1.0
    conv1_weight[0][6] = 16'h3C00; // 1.0
    
    conv1_weight[1][0] = 16'h3800; // 0.5
    conv1_weight[1][1] = 16'h3E00; // 1.5
    conv1_weight[1][2] = 16'hBE00; // -1.5
    conv1_weight[1][3] = 16'h0000; // 0.0
    conv1_weight[1][4] = 16'hBE00; // -1.5
    conv1_weight[1][5] = 16'h3E00; // 1.5
    conv1_weight[1][6] = 16'h3800; // 0.5
    
    conv1_weight[2][0] = 16'hB800; // -0.5
    conv1_weight[2][1] = 16'h3C00; // 1.0
    conv1_weight[2][2] = 16'hBC00; // -1.0
    conv1_weight[2][3] = 16'h4000; // 2.0
    conv1_weight[2][4] = 16'hBC00; // -1.0
    conv1_weight[2][5] = 16'h3C00; // 1.0
    conv1_weight[2][6] = 16'hB800; // -0.5
    
    // Conv2权重（8x3x5）
    // ... (从Python生成)
    
    // FC权重和偏置
    // ... (从Python生成)
end

//=============================================================================
// FP16运算函数
//=============================================================================

// FP16加法
function [15:0] fp16_add;
    input [15:0] a, b;
    reg sign_a, sign_b;
    reg [4:0] exp_a, exp_b;
    reg [10:0] mant_a, mant_b;
    reg [11:0] mant_a_ext, mant_b_ext;  // 包含隐含位
    reg [4:0] exp_diff;
    //reg [11:0] mant_aligned;
    reg [11:0] mant_sum;
    reg [5:0] exp_result;
    reg sign_result;
    reg [9:0] mant_result;
    reg[3:0] index;
    integer i;

    begin
        // 分解FP16
        sign_a = a[15];
        sign_b = b[15];
        exp_a = a[14:10] == 0 ? 5'b00001 : a[14:10];
        exp_b = b[14:10] == 0 ? 5'b00001 : b[14:10];
        mant_a = a[14:10] == 0 ? {1'b0, a[9:0]} : {1'b1, a[9:0]};  // 添加隐含位
        mant_b = b[14:10] == 0 ? {1'b0, b[9:0]} : {1'b1, b[9:0]};
        
        // 处理特殊情况（简化）
        if (a == 16'h0000 || a == 16'h8000) begin
            fp16_add = b;
        end else if (b == 16'h0000 || b == 16'h8000) begin
            fp16_add = a;
        end else begin
            // 对齐指数
            if (exp_a > exp_b) begin
                exp_diff = exp_a - exp_b;
                mant_b_ext = {0,(mant_b >> exp_diff)} + mant_b[exp_diff - 1];
                mant_a_ext = {0,mant_a};
                exp_result = exp_a;
            end else if (exp_b > exp_a) begin
                exp_diff = exp_b - exp_a;
                mant_a_ext = {0,(mant_a >> exp_diff)} + mant_a[exp_diff - 1];
                mant_b_ext = {0,mant_b};
                exp_result = exp_b;
            end
            else begin
                mant_b_ext = {0,mant_b};
                mant_a_ext = {0,mant_a};
                exp_result = exp_a;
                exp_diff = 0;
            end
            
            // 加法或减法
            if (sign_a == sign_b) begin
                mant_sum = mant_a_ext + mant_b_ext;
                sign_result = sign_a;
            end else begin
                if (mant_a_ext >= mant_b_ext) begin
                    mant_sum = mant_a_ext - mant_b_ext;
                    sign_result = sign_a;
                end else begin
                    mant_sum = mant_b_ext - mant_a_ext;
                    sign_result = sign_b;
                end
            end
            
            // 规范化 找到第一个1然后看是不是超过上下限了
            index = 0;
            for (i = 0; i < 11; i = i + 1) begin
            if (mant_sum[i]) begin
                index = i;
            end
            end
            // if(index == 11 && exp_result == 5'b1110)begin//超限保持最大值
            //     mant_result = 10'b1111_1111_11;
            // end
            // else if(index <= 9 && 10 - index >= exp_result)begin
            //     mant_sum = mant_sum << (exp_result - 1);
            //     mant_result = mant_sum[9:0];                
            //     exp_result = 0;
            // end
            // else if(index <= 9)begin
            //     mant_sum = mant_sum << (10 - index);
            //     mant_result = mant_sum[9:0];
            //     exp_result = exp_result - (10 - index);
            // end
                // if (mant_sum[11]) begin
                //     mant_sum = mant_sum >> 1;//应该是去掉第一个1
                //     exp_result = exp_result + 1;
                // end else if (!mant_sum[10] && exp_result > 0) begin
                //     mant_sum = mant_sum << 1;
                //     exp_result = exp_result - 1;
                // end
                
                // 组装结果
                // mant_result = mant_sum[9:0];
            if((index + exp_result >= 11) && (index - 10 + exp_result > 6'b011110))begin//超限保持最大值
                mant_result = 10'b1111_1111_11;
                exp_result = 6'b011110;
            end
            else if(index + exp_result < 11)begin//小数点前没有值
                if(exp_result >= 16) mant_sum = mant_sum << (exp_result - 16);//指数降到1
                else mant_sum = mant_sum >> (16 - exp_result);
                //mant_sum = mant_sum + {11{1'b0},mant_sum[9],10{1'b0}};//舍入
                if(index != 11 && mant_sum[11] && index + exp_result + 1 == 11)begin//若舍入还能进位导致指数到1
                    exp_result = 1;
                    mant_sum = mant_sum >> 1;
                end
                else exp_result = 0;
                mant_result = mant_sum[19:10];                                
            end
            // else if(index + exp_result < 26)begin//小于最小值
            //     mant_result = 0;                
            //     exp_result = 0;
            // end
            else begin
                if(index <= 20) mant_sum = mant_sum << (20 - index);
                else mant_sum = mant_sum >> (index - 20);
                mant_sum = mant_sum + {11{1'b0},mant_sum[9],10{1'b0}};//舍入
                if(mant_sum[11] && index + exp_result + 1 > 6'b011110)//若舍入还能进位导致指数到1
                    mant_result = 10'b1111_1111_11;
                    exp_result = 6'b011110;
                else begin
                    exp_result = exp_result - (35 - index) + mant_sum[11];
                    mant_sum = mant_sum >> mant_sum[11];
                end
                mant_result = mant_sum[19:10];
            end
            fp16_add = {sign_result, exp_result, mant_result};
        end
    end
endfunction

// FP16乘法
function [15:0] fp16_mult;
    input [15:0] a, b;
    reg sign_a, sign_b;
    reg [4:0] exp_a, exp_b;
    reg [10:0] mant_a, mant_b;
    reg [21:0] mant_product;
    reg [6:0] exp_result;
    reg sign_result;
    reg [9:0] mant_result;
    reg [6:0] index;
    integer i;
    begin
        // 分解FP16
        sign_a = a[15];
        sign_b = b[15];
        exp_a = a[14:10] == 0 ? 5'b00001 : a[14:10];
        exp_b = b[14:10] == 0 ? 5'b00001 : b[14:10];
        mant_a = a[14:10] == 0 ? {1'b0, a[9:0]} : {1'b1, a[9:0]};  // 添加隐含位
        mant_b = b[14:10] == 0 ? {1'b0, b[9:0]} : {1'b1, b[9:0]};
        
        // 处理零
        if (a == 16'h0000 || a == 16'h8000 || 
            b == 16'h0000 || b == 16'h8000) begin
            fp16_mult = 16'h0000;
        end else begin
            // 计算
            sign_result = sign_a ^ sign_b;
            exp_result = exp_a + exp_b;  // 先不减去偏置，防止下溢
            mant_product = mant_a * mant_b;
            
            index = 0;
            for (i = 0; i < 21; i = i + 1) begin
                if (mant_product[i]) begin
                    index = i;
                end
            end
            // 规范化
            // if (mant_product[21]) begin
            //     mant_product = mant_product >> 1;
            //     exp_result = exp_result + 1;
            // end
            if((index + exp_result >= 36) && (index - 35 + exp_result > 7'b0011110))begin//超限保持最大值
                mant_result = 10'b1111_1111_11;
                exp_result = 7'b0011110;
            end
            else if(index + exp_result < 36)begin//小数点前没有值
                if(exp_result >= 16) mant_product = mant_product << (exp_result - 16);//指数降到1
                else mant_product = mant_product >> (16 - exp_result);
                mant_product = mant_product + {11{1'b0},mant_product[9],10{1'b0}};//舍入
                if(index != 21 && mant_product[21] && index + exp_result + 1 == 36)begin//若舍入还能进位导致指数到1
                    exp_result = 1;
                    mant_product = mant_product >> 1;
                end
                else exp_result = 0;
                mant_result = mant_product[19:10];                                
            end
            // else if(index + exp_result < 26)begin//小于最小值
            //     mant_result = 0;                
            //     exp_result = 0;
            // end
            else begin
                if(index <= 20) mant_product = mant_product << (20 - index);
                else mant_product = mant_product >> (index - 20);
                mant_product = mant_product + {11{1'b0},mant_product[9],10{1'b0}};//舍入
                if(mant_product[21] && index + exp_result + 1 > 7'b0011110)begin//若舍入还能进位导致指数到1
                    mant_result = 10'b1111_1111_11;
                    exp_result = 7'b0011110;
                end
                else begin
                    exp_result = exp_result - (35 - index) + mant_product[21];
                    mant_product = mant_product >> mant_product[21];
                end
                mant_result = mant_product[19:10];
                
            end
            
            // 组装结果
            //mant_result = mant_product[19:10];
            fp16_mult = {sign_result, exp_result[4:0], mant_result};
        end
    end
endfunction

// FP16比较（大于）
function fp16_gt;
    input [15:0] a, b;
    reg sign_a, sign_b;
    begin
        sign_a = a[15];
        sign_b = b[15];
        
        if (sign_a != sign_b) begin
            fp16_gt = !sign_a;  // 正数大于负数
        end else if (sign_a == 0) begin
            fp16_gt = (a > b);  // 都是正数
        end else begin
            fp16_gt = (a < b);  // 都是负数
        end
    end
endfunction

// FP16 ReLU
function [15:0] fp16_relu;
    input [15:0] x;
    begin
        if (x[15])  // 负数
            fp16_relu = 16'h0000;
        else
            fp16_relu = x;
    end
endfunction

// FP16最大值
function [15:0] fp16_max;
    input [15:0] a, b;
    begin
        if (fp16_gt(a, b))
            fp16_max = a;
        else
            fp16_max = b;
    end
endfunction

//=============================================================================
// 中间缓冲区
//=============================================================================
reg [DATA_WIDTH-1:0] conv1_out [0:2][0:SEQ_LEN-1];
reg [DATA_WIDTH-1:0] relu1_out [0:2][0:SEQ_LEN-1];
reg [DATA_WIDTH-1:0] pool1_out [0:2][0:9];

reg [DATA_WIDTH-1:0] conv2_out [0:7][0:9];
reg [DATA_WIDTH-1:0] relu2_out [0:7][0:9];
reg [DATA_WIDTH-1:0] pool2_out [0:7][0:4];

reg [DATA_WIDTH-1:0] adaptpool_out [0:7][0:2];
reg [DATA_WIDTH-1:0] flatten_out [0:23];

reg [DATA_WIDTH-1:0] logits [0:1];

//=============================================================================
// 状态机
//=============================================================================
localparam [3:0] IDLE         = 4'd0;
localparam [3:0] CONV1_CALC   = 4'd1;
localparam [3:0] RELU1_CALC   = 4'd2;
localparam [3:0] POOL1_CALC   = 4'd3;
localparam [3:0] CONV2_CALC   = 4'd4;
localparam [3:0] RELU2_CALC   = 4'd5;
localparam [3:0] POOL2_CALC   = 4'd6;
localparam [3:0] ADAPTPOOL    = 4'd7;
localparam [3:0] FLATTEN_CALC = 4'd8;
localparam [3:0] FC_CALC      = 4'd9;
localparam [3:0] COMPARE      = 4'd10;
localparam [3:0] DONE_STATE   = 4'd11;

reg [3:0] state, next_state;
reg [7:0] counter;
reg [7:0] i, j, k, m;
reg [15:0] temp_sum;
reg [15:0] temp_max;

//=============================================================================
// 状态机转换
//=============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state <= IDLE;
    else
        state <= next_state;
end

always @(*) begin
    next_state = state;
    case (state)
        IDLE:         if (start) next_state = CONV1_CALC;
        CONV1_CALC:   if (counter == 3) next_state = RELU1_CALC;
        RELU1_CALC:   if (counter == 3) next_state = POOL1_CALC;
        POOL1_CALC:   if (counter == 3) next_state = CONV2_CALC;
        CONV2_CALC:   if (counter == 8) next_state = RELU2_CALC;
        RELU2_CALC:   if (counter == 8) next_state = POOL2_CALC;
        POOL2_CALC:   if (counter == 8) next_state = ADAPTPOOL;
        ADAPTPOOL:    if (counter == 8) next_state = FLATTEN_CALC;
        FLATTEN_CALC: next_state = FC_CALC;
        FC_CALC:      if (counter == 2) next_state = COMPARE;
        COMPARE:      next_state = DONE_STATE;
        DONE_STATE:   next_state = IDLE;
        default:      next_state = IDLE;
    endcase
end

//=============================================================================
// 主计算逻辑
//=============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 0;
        counter <= 0;
        prediction <= 0;
        for (i = 0; i < 2; i = i + 1) begin
            logits[i] <= 0;
            logits_out[i] <= 0;
        end
    end else begin
        case (state)
            IDLE: begin
                done <= 0;
                counter <= 0;
            end
            
            // Conv1层
            CONV1_CALC: begin
                if (counter < 3) begin
                    for (j = 0; j < SEQ_LEN; j = j + 1) begin
                        temp_sum = 16'h0000;
                        for (k = 0; k < 7; k = k + 1) begin
                            if (j + k >= 3 && j + k < SEQ_LEN + 3) begin
                                m = j + k - 3;
                                temp_sum = fp16_add(temp_sum, 
                                    fp16_mult(conv1_weight[counter][k], rr_data[m]));
                            end
                        end
                        conv1_out[counter][j] = temp_sum;
                    end
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // ReLU1层
            RELU1_CALC: begin
                if (counter < 3) begin
                    for (j = 0; j < SEQ_LEN; j = j + 1) begin
                        relu1_out[counter][j] = fp16_relu(conv1_out[counter][j]);
                    end
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // Pool1层
            POOL1_CALC: begin
                if (counter < 3) begin
                    for (j = 0; j < SEQ_LEN/2; j = j + 1) begin
                        m = j * 2;
                        temp_max = relu1_out[counter][m];
                        if (m > 0)
                            temp_max = fp16_max(temp_max, relu1_out[counter][m-1]);
                        else
                            temp_max = fp16_max(temp_max, 16'h0000);
                        if (m < SEQ_LEN - 1)
                            temp_max = fp16_max(temp_max, relu1_out[counter][m+1]);
                        else
                            temp_max = fp16_max(temp_max, 16'h0000);
                        pool1_out[counter][j] = temp_max;
                    end
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // Conv2层
            CONV2_CALC: begin
                if (counter < 8) begin
                    for (j = 0; j < SEQ_LEN/2; j = j + 1) begin
                        temp_sum = 16'h0000;
                        for (k = 0; k < 3; k = k + 1) begin
                            for (m = 0; m < 5; m = m + 1) begin
                                if (j + m >= 2 && j + m < SEQ_LEN/2 + 2) begin
                                    temp_sum = fp16_add(temp_sum,
                                        fp16_mult(conv2_weight[counter][k][m], 
                                                 pool1_out[k][j+m-2]));
                                end
                            end
                        end
                        conv2_out[counter][j] = temp_sum;
                    end
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // ReLU2层
            RELU2_CALC: begin
                if (counter < 8) begin
                    for (j = 0; j < SEQ_LEN/2; j = j + 1) begin
                        relu2_out[counter][j] = fp16_relu(conv2_out[counter][j]);
                    end
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // Pool2层
            POOL2_CALC: begin
                if (counter < 8) begin
                    for (j = 0; j < SEQ_LEN/4; j = j + 1) begin
                        m = j * 2;
                        temp_max = relu2_out[counter][m];
                        if (m > 0)
                            temp_max = fp16_max(temp_max, relu2_out[counter][m-1]);
                        else
                            temp_max = fp16_max(temp_max, 16'h0000);
                        if (m < SEQ_LEN/2 - 1)
                            temp_max = fp16_max(temp_max, relu2_out[counter][m+1]);
                        else
                            temp_max = fp16_max(temp_max, 16'h0000);
                        pool2_out[counter][j] = temp_max;
                    end
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // AdaptiveAvgPool层
            ADAPTPOOL: begin
                if (counter < 8) begin
                    // 简化：直接相加后除以2
                    temp_sum = fp16_add(pool2_out[counter][0], pool2_out[counter][1]);
                    adaptpool_out[counter][0] = {temp_sum[15], temp_sum[14:10] - 5'b00001, temp_sum[9:0]};
                    
                    temp_sum = fp16_add(pool2_out[counter][2], pool2_out[counter][3]);
                    adaptpool_out[counter][1] = {temp_sum[15], temp_sum[14:10] - 5'b00001, temp_sum[9:0]};
                    
                    adaptpool_out[counter][2] = pool2_out[counter][4];
                    
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // Flatten层
            FLATTEN_CALC: begin
                for (i = 0; i < 8; i = i + 1) begin
                    for (j = 0; j < POOL_LEN; j = j + 1) begin
                        flatten_out[i*POOL_LEN + j] = adaptpool_out[i][j];
                    end
                end
                counter <= 0;
            end
            
            // FC层
            FC_CALC: begin
                if (counter < 2) begin
                    temp_sum = fc_bias[counter];
                    for (j = 0; j < 24; j = j + 1) begin
                        temp_sum = fp16_add(temp_sum,
                            fp16_mult(fc_weight[counter][j], flatten_out[j]));
                    end
                    logits[counter] = temp_sum;
                    logits_out[counter] = temp_sum;
                    counter <= counter + 1;
                end else begin
                    counter <= 0;
                end
            end
            
            // 比较
            COMPARE: begin
                if (fp16_gt(logits[0], logits[1]))
                    prediction <= 1'b0;
                else
                    prediction <= 1'b1;
                counter <= 0;
            end
            
            // 完成
            DONE_STATE: begin
                done <= 1;
                counter <= 0;
            end
            
            default: begin
                counter <= 0;
            end
        endcase
    end
end

endmodule