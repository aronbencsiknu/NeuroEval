`include "specs.v"

//integer num_elements = `NUM_PACKETS_P_INJ;



integer counterRecieved = 0;
integer counterSent = 0;

//integer ts_len = 50;

module tb_mt_stage;
    //import my_pkg::*;
	logic reset;

	logic ReqInE, AckInE, ReqOutE, AckOutE; logic [0:`DATA_W-1] DataInE, DataOutE; // Port E
	logic ReqInN, AckInN, ReqOutN, AckOutN; logic [0:`DATA_W-1] DataInN, DataOutN; // Port N
	logic ReqInW, AckInW, ReqOutW, AckOutW; logic [0:`DATA_W-1] DataInW, DataOutW; // Port W
	logic ReqInS, AckInS, ReqOutS, AckOutS; logic [0:`DATA_W-1] DataInS, DataOutS; // Port S
	logic ReqInL1, AckInL1, ReqOutL1, AckOutL1; logic [0:`DATA_W-1] DataInL1, DataOutL1; // Port L1

    logic everything_delivered;
    logic finished_sending;
    logic sync;
    
    logic init_done;

    logic finish;
    logic [`DATA_W-1:0] expandedPacketsList[$];

    int fruits[`NUM_NEURONS][`ADDR_W][`CORE_CAPACITY];
    int veggies[`NUM_NEURONS][`ADDR_W][`CORE_CAPACITY];

    // Definitions for directions
    typedef enum {EAST, NORTH, WEST, SOUTH, L1} direction_t;

    // Define arrays for each direction
    logic [`DATA_W-1:0] packetListE[`NUM_PACKETS_P_INJ];

    logic [`DATA_W-1:0] testlist[`NUM_PACKETS_P_INJ];

    logic [`DATA_W-1:0] packetListN[`NUM_PACKETS_P_INJ];

    logic [`DATA_W-1:0] packetListW[`NUM_PACKETS_P_INJ];

    logic [`DATA_W-1:0] packetListS[`NUM_PACKETS_P_INJ];

    logic [`DATA_W-1:0] packetListL1[`NUM_PACKETS_P_INJ];

    integer packetCounterE = 0;
    integer packetCounterN = 0;
    integer packetCounterW = 0;
    integer packetCounterS = 0;
    integer packetCounterL1 = 0;

    integer packetLimitE = 0;
    integer packetLimitN = 0;
    integer packetLimitW = 0;
    integer packetLimitS = 0;
    integer packetLimitL1 = 0;

    integer counter_reset;
    logic empty;
    logic packet_expand_done = 0;

    //logic [14:0] tb_mt_stage.expandedPacketsList[$]; // Assuming 50 is enough to hold all expanded addresses
    integer i, j, index = 0;

    task expand_address_list(
    input reg [`DATA_W-1:0] packet_list[`NUM_PACKETS_P_INJ], 
    //output reg [0:`DATA_W-1] out_list[],
    input integer limit,
    input integer core_id
    );
    //integer i, j;
    reg [`ADDR_W-1:0] address;
    reg [`MSG_W-1:0] message;
    //logic [`MSG_W-1:0] message;
    logic [(`MSG_W/2)-1:0] message_first_half;
    logic [(`MSG_W/2)-1:0] message_second_half;
        for (i = 0; i < limit; i++) begin
            
            message = packet_list[i][`ADDR_W+`MSG_W-1:`ADDR_W];
            message_first_half = message[`MSG_W-1:`MSG_W/2];
            message_second_half = message[(`MSG_W/2)-1:0];

            address = packet_list[i][`ADDR_W-1:0];

            // Extract the last ADDR_W bits for the address
            //$display("entire: %b", packet_list[i]);
            //$display("First half (integer): %0d", message_first_half);
            //$display("Second half (integer): %0d", message_second_half);
            //$display("ENTIRE ADDRESS: %b", address);
            for (j = 0; j < `ADDR_W; j++) begin
                    if (address[j]) begin
                        //$display("PACKET: core %d:%d:%b", core_id, `ADDR_W-1-j, 5'b00001 << j);
                        
			            tb_mt_stage.expandedPacketsList = {tb_mt_stage.expandedPacketsList, {message, 5'b00001 << j}}; // Shift 1 to the position of each set bit, concatenate with message
                        
                        //tb_mt_stage.fruits[0][0][0] += 1;
                        tb_mt_stage.fruits[message_first_half][`ADDR_W-1-j][message_second_half] += 1;
                    end
            end
        end
    endtask

    initial begin

        finished_sending = 0;
        everything_delivered = 0;
        counter_reset = 0;
        empty = 0;

        reset = 1'b0; // Reset is active (low-active reset)
        #500;         // Delay to ensure reset is perceived as active
        reset = 1'b1; // Deactivate reset

        // Initialize Request and Data for portE
        ReqInE = 0;
        DataInE = 15'b0;

        // Initialize Request and Data for portN
        ReqInN = 0;
        DataInN = 15'b0;
        
        // Initialize Request and Data for portW
        ReqInW = 0;
        DataInW = 15'b0; 
    
        // Initialize Request and Data for portS
        ReqInS = 0;
        DataInS = 15'b0;

        // Initialize Request and Data for portL1
        ReqInL1 = 0;
        DataInL1 = 15'b0;
        
        index = 0;

        
        init_done = 1;
         
        
        //@(posedge everything_delivered); // Wait for a positive edge
        
        //#10; 
        @(posedge finish);
        $display("RECIEVED: %d", counterRecieved);
        $display("SENT: %d", counterSent);

        //Display data for L1 direction
        // $display("Data for L1 direction:");
        // foreach (tb_mt_stage.expandedPacketsList[i]) begin
        //     $display("Address: %b",tb_mt_stage.expandedPacketsList[i]);
        // end

        //$finish; // End of simulation

  	end

    switch_5x5_XY sw (
        // IO ports for portE (asynchronous)
        .ReqInE(ReqInE),
        .DataInE(DataInE),
        .AckInE(AckInE),
        .ReqOutE(ReqOutE),
        .DataOutE(DataOutE),
        .AckOutE(AckOutE),

        // IO ports for portN (asynchronous)
        .ReqInN(ReqInN),
        .DataInN(DataInN),
        .AckInN(AckInN),
        .ReqOutN(ReqOutN),
        .DataOutN(DataOutN),
        .AckOutN(AckOutN),

        // IO ports for portW (asynchronous)
        .ReqInW(ReqInW),
        .DataInW(DataInW),
        .AckInW(AckInW),
        .ReqOutW(ReqOutW),
        .DataOutW(DataOutW),
        .AckOutW(AckOutW),

        // IO ports for portS (asynchronous)
        .ReqInS(ReqInS),
        .DataInS(DataInS),
        .AckInS(AckInS),
        .ReqOutS(ReqOutS),
        .DataOutS(DataOutS),
        .AckOutS(AckOutS),

        // IO ports for portL1 (asynchronous)
        .ReqInL1(ReqInL1),
        .DataInL1(DataInL1),
        .AckInL1(AckInL1),
        .ReqOutL1(ReqOutL1),
        .DataOutL1(DataOutL1),
        .AckOutL1(AckOutL1),

        // Additional IOs
        .SID(`SID),
        .E_MASK(`E_MASK),
        .W_MASK(`W_MASK),
        .N_MASK(`N_MASK),
        .S_MASK(`S_MASK),

        .rst(reset)
	);
	
	// Provide Ack signal for the Out ports
  	back_ack ba_E(.req(ReqOutE), .ack(AckOutE), .reset(reset));
	back_ack ba_N(.req(ReqOutN), .ack(AckOutN), .reset(reset));
	back_ack ba_W(.req(ReqOutW), .ack(AckOutW), .reset(reset));
	back_ack ba_S(.req(ReqOutS), .ack(AckOutS), .reset(reset));
	back_ack ba_L1(.req(ReqOutL1), .ack(AckOutL1), .reset(reset));

    expand_packets ep(.counter_reset(counter_reset));
    signal_finished_sending ste();
    empty_expand_packets eep();
    //sync_remaining sr();

    // Stimulus for East direction
    stimulus fs_E(
        .input_req(ReqInE),
        .ack(AckInE),
        .packet_list(packetListE),
        .stimulus(DataInE),
        .output_req(ReqInE),
        .counter(packetCounterE),
        .counter_out(packetCounterE),
        .limit(packetLimitE)
    );

    // Stimulus for North direction
    stimulus fs_N(
        .input_req(ReqInN),
        .ack(AckInN),
        .packet_list(packetListN),
        .stimulus(DataInN),
        .output_req(ReqInN),
        .counter(packetCounterN),
        .counter_out(packetCounterN),
        .limit(packetLimitN)
    );

    // Stimulus for West direction
    stimulus fs_W(
        .input_req(ReqInW),
        .ack(AckInW),
        .packet_list(packetListW),
        .stimulus(DataInW),
        .output_req(ReqInW),
        .counter(packetCounterW),
        .counter_out(packetCounterW),
        .limit(packetLimitW)
    );

    // Stimulus for South direction
    stimulus fs_S(
        .input_req(ReqInS),
        .ack(AckInS),
        .packet_list(packetListS),
        .stimulus(DataInS),
        .output_req(ReqInS),
        .counter(packetCounterS),
        .counter_out(packetCounterS),
        .limit(packetLimitS)
    );

    // Stimulus for L1 direction
    stimulus fs_L1(
        .input_req(ReqInL1),
        .ack(AckInL1),
        .packet_list(packetListL1),
        .stimulus(DataInL1),
        .output_req(ReqInL1),
        .counter(packetCounterL1),
        .counter_out(packetCounterL1),
        .limit(packetLimitL1)
    );

    // Check which packets arrived correctly from the predefined list
    check_packet_transit cpt_E(.dataOut(DataOutE), .req(ReqOutE), .finished_sending_in(finished_sending), .finished_sending_out(finished_sending), .core_id(`EAST));
    check_packet_transit cpt_N(.dataOut(DataOutN), .req(ReqOutN), .finished_sending_in(finished_sending), .finished_sending_out(finished_sending), .core_id(`NORTH));
    check_packet_transit cpt_W(.dataOut(DataOutW), .req(ReqOutW), .finished_sending_in(finished_sending), .finished_sending_out(finished_sending), .core_id(`WEST));
    check_packet_transit cpt_S(.dataOut(DataOutS), .req(ReqOutS), .finished_sending_in(finished_sending), .finished_sending_out(finished_sending), .core_id(`SOUTH));
    check_packet_transit cpt_L1(.dataOut(DataOutL1), .req(ReqOutL1), .finished_sending_in(finished_sending), .finished_sending_out(finished_sending), .core_id(`L1));

endmodule

module expand_packets(input integer counter_reset);

 always @(counter_reset) begin
    begin
        //$display("HERE-");
        tb_mt_stage.packet_expand_done = 0;
        //tb_mt_stage.expandedPacketsList.delete();
        expand_address_list(tb_mt_stage.packetListE, tb_mt_stage.packetLimitE, `EAST);
        expand_address_list(tb_mt_stage.packetListN, tb_mt_stage.packetLimitN, `NORTH);
        expand_address_list(tb_mt_stage.packetListW, tb_mt_stage.packetLimitW, `WEST);
        expand_address_list(tb_mt_stage.packetListS, tb_mt_stage.packetLimitS, `SOUTH);
        expand_address_list(tb_mt_stage.packetListL1, tb_mt_stage.packetLimitL1, `L1);
        // foreach(tb_mt_stage.fruits[i]) begin
        //     tb_mt_stage.fruits[i] = 0;
        // end
        // foreach(tb_mt_stage.expandedPacketsList[i]) begin
        //     tb_mt_stage.fruits[i] = tb_mt_stage.expandedPacketsList[i];
        // end
        tb_mt_stage.packet_expand_done = 1;
    end
 end

endmodule

module empty_expand_packets();
 always @(tb_mt_stage.empty) begin
    //packet_expand_done = 0;
    tb_mt_stage.expandedPacketsList.delete();
    for (int i = 0; i < `NUM_NEURONS; i++) begin
            for (int j = 0; j < `ADDR_W; j++) begin
                for (int k = 0; k < `CORE_CAPACITY; k++) begin
                    tb_mt_stage.fruits[i][j][k] = 0;
                    tb_mt_stage.veggies[i][j][k] = 0;
                end
            end
        end
    tb_mt_stage.empty = 0;
 end
endmodule

// module sync_remaining();
//     always @(tb_mt_stage.sync) begin
//         packet_expand_done = 0;
//         foreach(tb_mt_stage.fruits[i]) begin
//             tb_mt_stage.fruits[i] = 0;
//         end
//         foreach(tb_mt_stage.expandedPacketsList[i]) begin
//             tb_mt_stage.fruits[i] = tb_mt_stage.expandedPacketsList[i];
//         end
//     end
// endmodule
module signal_finished_sending();
    always @(tb_mt_stage.packetCounterE,
        tb_mt_stage.packetCounterN,
        tb_mt_stage.packetCounterW,
        tb_mt_stage.packetCounterS,
        tb_mt_stage.packetCounterL1) begin

        if(tb_mt_stage.packetCounterE == tb_mt_stage.packetLimitE &&
        tb_mt_stage.packetCounterN == tb_mt_stage.packetLimitN  &&
        tb_mt_stage.packetCounterW == tb_mt_stage.packetLimitW  &&
        tb_mt_stage.packetCounterS == tb_mt_stage.packetLimitS  &&
        tb_mt_stage.packetCounterL1 == tb_mt_stage.packetLimitL1 ) begin

                // $display("Counter is %d", tb_mt_stage.packetCounterE);
                // $display("Counter is %d", tb_mt_stage.packetCounterN);
                // $display("Counter is %d", tb_mt_stage.packetCounterW);
                // $display("Counter is %d", tb_mt_stage.packetCounterS);
                // $display("Counter is %d", tb_mt_stage.packetCounterL1);

                tb_mt_stage.finished_sending++;
            end
        end
    
endmodule

module stimulus(
    input input_req,
    input ack,
    input reg [`DATA_W-1:0] packet_list[],
    input integer counter, // Input integer
    output integer counter_out, // Output integer to pass the updated counter
    output reg output_req,
    output reg [`DATA_W-1:0] stimulus,
    input integer limit
);
  
    initial begin
        counter_out = 0;
    end
    always @(ack, input_req, tb_mt_stage.packet_expand_done) begin
        
        //$display("Data is %b", packet_list[counter]);
        //$display("OUT %d", counter);
        
        if (ack == input_req && tb_mt_stage.packet_expand_done == 1 && counter < limit) begin
            counterSent += 1;
            //$display("IN");
            
       	    //#150 stimulus <= {data_list[counter], address_list[counter]};
            #150 stimulus <= packet_list[counter];
            
            #150 output_req <= ~input_req; //METASTABILITY test
            //$display("Data SENT %b -- %d", packet_list[counter], $time);
            //$display();
	        counter_out = counter + 1;  // Increment the counter
        end
        else begin
            counter_out = counter;
        end
        
    end
endmodule

module check_packet_transit(
	input logic [`DATA_W-1:0] dataOut,
    input req,
    input logic finished_sending_in,
    output logic finished_sending_out,
    input integer core_id
);

        //reg [`ADDR_W-1:0] address;
        reg [`MSG_W-1:0] message;
        logic [(`MSG_W/2)-1:0] message_first_half;
        logic [(`MSG_W/2)-1:0] message_second_half;
	
	always @(req) begin
		automatic int index = -1;
        counterRecieved = counterRecieved + 1;
            
        message = dataOut[`ADDR_W+`MSG_W-1:`ADDR_W];
        //$display("Data OUT- %b -- %d", dataOut, $time);
        message_first_half = message[`MSG_W-1:`MSG_W/2];
        message_second_half = message[(`MSG_W/2)-1:0];
        
		foreach(tb_mt_stage.expandedPacketsList[i]) begin
			if (tb_mt_stage.expandedPacketsList[i] == dataOut) begin
                		index = i;
                		break;
			end
        end
        //$display("Size of list is: %d", tb_mt_stage.expandedPacketsList.size());
        if (index != -1) begin
	        //$display("############### Success: Value %b was found in the packet list. %d", dataOut, $size(tb_mt_stage.expandedPacketsList));
            tb_mt_stage.expandedPacketsList.delete(index);  // Remove the item if found
            tb_mt_stage.fruits[message_first_half][core_id][message_second_half] -= 1;

        end else begin
            //$display("Value %b not found in the packet list.", dataOut);
            tb_mt_stage.veggies[message_first_half][core_id][message_second_half] += 1;
        end
        if ($size(tb_mt_stage.expandedPacketsList) <= 0) begin
            //$display("DELIVERED EVERYTHING BITCH %d %d", tb_mt_stage.counter_reset, $size(tb_mt_stage.expandedPacketsList));
                tb_mt_stage.everything_delivered++;
                
        end

        //packetsToDeliver_o <= packetsToDeliver;

	end

// check if recieved packet is in list defined above and remove. At the end, examine the list and see if any remains.

endmodule

// Provide ack signal to the last mousetrap stage
module back_ack(
	input req, 
	input reset, 
	output reg ack
);

	always@(req) begin
    	if(~reset)
      		ack<= 0;
  		else
    		#10 ack<= req;
	end
endmodule