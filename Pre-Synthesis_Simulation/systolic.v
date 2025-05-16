module systolic#(
    parameter ARRAY_SIZE = 8,
    parameter SRAM_DATA_WIDTH = 32,
    parameter DATA_WIDTH = 8,
    // --- 显式添加累加深度参数 ---
    parameter K_ACCUM_DEPTH = 8   // 用户可配置的累加深度，默认为原始行为 (K=8)
)
(
    input clk,
    input srstn,
    input alu_start,
    input [8:0] cycle_num,     // 如果K_ACCUM_DEPTH非常大，可能需要更宽

    input [SRAM_DATA_WIDTH-1:0] sram_rdata_w0,
    input [SRAM_DATA_WIDTH-1:0] sram_rdata_w1,

    input [SRAM_DATA_WIDTH-1:0] sram_rdata_d0,
    input [SRAM_DATA_WIDTH-1:0] sram_rdata_d1,

    input [5:0] matrix_index,
    // --- 根据新的OUTCOME_WIDTH调整输出总线宽度 ---
    output reg signed [(ARRAY_SIZE * (DATA_WIDTH + DATA_WIDTH + ((K_ACCUM_DEPTH == 1) ? 0 : $clog2(K_ACCUM_DEPTH)) + 1)) - 1:0] mul_outcome
);

localparam ITEMS_PER_SRAM_WORD = SRAM_DATA_WIDTH / DATA_WIDTH;
// --- 根据 K_ACCUM_DEPTH 动态计算累加器位宽 ---
// K=1时，headroom为0；否则为log2(K)
localparam ACCUMULATOR_HEADROOM = (K_ACCUM_DEPTH == 1) ? 0 : $clog2(K_ACCUM_DEPTH);
// 乘积位宽 + headroom + 1位额外buffer/符号对齐
localparam OUTCOME_WIDTH = DATA_WIDTH + DATA_WIDTH + ACCUMULATOR_HEADROOM + 1;

// --- 基于 K_ACCUM_DEPTH 重新定义控制逻辑参数 ---
// INITIAL_CTRL_OFFSET: 控制逻辑开始影响PE(0,0)的初始周期偏移，
//                      可以基于阵列填充时间。与原FIRST_OUT概念类似。
localparam INITIAL_CTRL_OFFSET = ARRAY_SIZE + 1;

// WAVEFRONT1_START_OFFSET: 第一个“复位累加器”波前的起始偏移
localparam WAVEFRONT1_START_OFFSET = INITIAL_CTRL_OFFSET;
// WAVEFRONT2_START_OFFSET: 第二个交错的“复位累加器”波前的起始偏移。
//                          它与第一个波前相差 K_ACCUM_DEPTH 个周期，
//                          这个差值直接定义了累加深度K。
localparam WAVEFRONT2_START_OFFSET = INITIAL_CTRL_OFFSET + K_ACCUM_DEPTH;

// WAVEFRONT_MODULO: 控制每个波前重复的周期。
//                   设置为 K_ACCUM_DEPTH * 2 可以确保两个波前正确交错，
//                   从而使得每个PE的累加器复位间隔为 K_ACCUM_DEPTH。
//                   处理 K_ACCUM_DEPTH=0 的情况以避免模0（尽管K通常>=1）。
localparam WAVEFRONT_MODULO = (K_ACCUM_DEPTH == 0) ? 1 : (K_ACCUM_DEPTH * 2);


// --- 内部寄存器和线网声明 (使用新的 OUTCOME_WIDTH) ---
reg signed [OUTCOME_WIDTH-1:0] matrix_mul_2D [0:ARRAY_SIZE-1] [0:ARRAY_SIZE-1];
reg signed [OUTCOME_WIDTH-1:0] matrix_mul_2D_nx [0:ARRAY_SIZE-1] [0:ARRAY_SIZE-1];
reg signed [DATA_WIDTH-1:0] data_queue [0:ARRAY_SIZE-1] [0:ARRAY_SIZE-1];
reg signed [DATA_WIDTH-1:0] weight_queue [0:ARRAY_SIZE-1] [0:ARRAY_SIZE-1];

reg signed [DATA_WIDTH+DATA_WIDTH-1:0] mul_result; // 乘积结果位宽不变

reg [5:0] upper_bound;
reg [5:0] lower_bound;

integer i,j;


//------data, weight------ (这部分逻辑与累加深度K的设置无直接关系，保持不变)
always@(posedge clk) begin
    if(~srstn) begin
        for(i=0; i<ARRAY_SIZE; i=i+1) begin
            for(j=0; j<ARRAY_SIZE; j=j+1) begin
                weight_queue[i][j] <= 0;
                data_queue[i][j]   <= 0;
            end
        end
    end
    else begin
        if(alu_start) begin
            //weight shifting(a0)
            // 使用参数化DATA_WIDTH处理数据加载，假设SRAM_DATA_WIDTH是DATA_WIDTH的整数倍
            // 这里假设每32位SRAM输入包含 32/DATA_WIDTH 个数据项
            for(i=0; i < ITEMS_PER_SRAM_WORD && i < ARRAY_SIZE ; i=i+1) begin // 确保不越界 ARRAY_SIZE
                 // sram_rdata_w0 和 w1 填充 weight_queue[0] 的不同部分
                 // 假设填充 weight_queue[0][0] 到 weight_queue[0][ARRAY_SIZE-1]
                 // 如果 ITEMS_PER_SRAM_WORD * 2 < ARRAY_SIZE, 需要更多SRAM输入或调整逻辑
                if (i < ARRAY_SIZE)
                    weight_queue[0][i] <= sram_rdata_w0[SRAM_DATA_WIDTH - DATA_WIDTH*i -1 -: DATA_WIDTH];
                if (ITEMS_PER_SRAM_WORD + i < ARRAY_SIZE) // 确保不越界
                    weight_queue[0][ITEMS_PER_SRAM_WORD + i] <= sram_rdata_w1[SRAM_DATA_WIDTH - DATA_WIDTH*i -1 -: DATA_WIDTH];
            end
            
            for(i=1; i<ARRAY_SIZE; i=i+1) 
                for(j=0; j<ARRAY_SIZE; j=j+1) 
                    weight_queue[i][j] <= weight_queue[i-1][j];
                
            //data shifting(b0)
            for(i=0; i < ITEMS_PER_SRAM_WORD && i < ARRAY_SIZE; i=i+1) begin // 确保不越界 ARRAY_SIZE
                if (i < ARRAY_SIZE)
                    data_queue[i][0] <= sram_rdata_d0[SRAM_DATA_WIDTH - DATA_WIDTH*i -1 -: DATA_WIDTH];
                if (ITEMS_PER_SRAM_WORD + i < ARRAY_SIZE) // 确保不越界
                    data_queue[ITEMS_PER_SRAM_WORD + i][0] <= sram_rdata_d1[SRAM_DATA_WIDTH - DATA_WIDTH*i -1 -: DATA_WIDTH];
            end
            
            for(i=0; i<ARRAY_SIZE; i=i+1) 
                for(j=1; j<ARRAY_SIZE; j=j+1) 
                    data_queue[i][j] <= data_queue[i][j-1];
        end
    end
end

//-------multiplication unit------------
always@(posedge clk) begin
    if(~srstn) begin
        for(i=0; i<ARRAY_SIZE; i=i+1) 
            for(j=0; j<ARRAY_SIZE; j=j+1)  
                matrix_mul_2D[i][j] <= 0;
    end
    else begin
        for(i=0; i<ARRAY_SIZE; i=i+1) 
            for(j=0; j<ARRAY_SIZE; j=j+1) 
                matrix_mul_2D[i][j] <= matrix_mul_2D_nx[i][j];
    end
end

always@(*) begin
    // 默认赋值以避免不必要的锁存器推断
    for(i=0; i<ARRAY_SIZE; i=i+1) begin
        for(j=0; j<ARRAY_SIZE; j=j+1) begin
            matrix_mul_2D_nx[i][j] = matrix_mul_2D[i][j];
        end
    end
    mul_result = 0; // 默认乘积结果

    if(alu_start) begin
        for(i=0; i<ARRAY_SIZE; i=i+1) begin
            for(j=0; j<ARRAY_SIZE; j=j+1) begin
                mul_result = weight_queue[i][j] * data_queue[i][j];

                // --- 使用新的控制参数修改条件判断 ---
                // 条件：PE(i,j) 开始计算新输出元素的第一个乘积 (复位累加器)
                if ( (cycle_num >= WAVEFRONT1_START_OFFSET && (i+j) == (cycle_num - WAVEFRONT1_START_OFFSET) % WAVEFRONT_MODULO) || 
                     (cycle_num >= WAVEFRONT2_START_OFFSET && (i+j) == (cycle_num - WAVEFRONT2_START_OFFSET) % WAVEFRONT_MODULO) ) begin
                    // 第一个乘积，直接赋值，注意符号扩展到新的 OUTCOME_WIDTH
                    matrix_mul_2D_nx[i][j] = {{OUTCOME_WIDTH - (DATA_WIDTH*2){mul_result[DATA_WIDTH*2-1]}}, mul_result};
                end
                // 条件：PE(i,j) 进行累加 (仅当K > 1时)
                // (i+j) <= (cycle_num-1) 是一个基本的数据到达和有效性检查
                else if ( K_ACCUM_DEPTH > 1 && cycle_num >= 1 && (i+j) <= (cycle_num-1) ) begin
                    // 累加后续乘积，注意符号扩展
                    matrix_mul_2D_nx[i][j] = matrix_mul_2D[i][j] + {{OUTCOME_WIDTH - (DATA_WIDTH*2){mul_result[DATA_WIDTH*2-1]}}, mul_result};
                end
                // 其他情况 (例如 K=1 且不是“第一个乘积”的那个周期，或者 cycle_num 太小等)
                // 已经在循环开始时通过 matrix_mul_2D_nx[i][j] = matrix_mul_2D[i][j]; 保持旧值。
            end
        end
    end
    // 如果 alu_start 为低，matrix_mul_2D_nx 保持旧值，mul_result 为0 (已默认设置)。
end

//------output data: mul_outcome(indexed by matrix_index)------
// (输出逻辑本身不变，但 mul_outcome 的总宽度已在模块端口处更新)
always@(*) begin
    if(matrix_index < ARRAY_SIZE) begin
        upper_bound = matrix_index;
        lower_bound = matrix_index + ARRAY_SIZE;
    end
    else begin
        upper_bound = matrix_index - ARRAY_SIZE;
        lower_bound = matrix_index;
    end

    mul_outcome = 'bz; // 推荐为输出赋一个明确的默认值

    for(i=0; i<ARRAY_SIZE; i=i+1) begin
        for(j=0; j<ARRAY_SIZE-i; j=j+1) begin
            if(i+j == upper_bound)
                mul_outcome[(i*OUTCOME_WIDTH) +: OUTCOME_WIDTH] = matrix_mul_2D[i][j];
        end
    end

    for(i=1; i<ARRAY_SIZE; i=i+1) begin
        for(j=ARRAY_SIZE-i; j<ARRAY_SIZE; j=j+1) begin
            if(i+j == lower_bound)
                mul_outcome[(i*OUTCOME_WIDTH) +: OUTCOME_WIDTH] = matrix_mul_2D[i][j];
        end
    end
end

endmodule