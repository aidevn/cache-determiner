#define SYCL_DISABLE_DEPRECATION_WARNINGS
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/ranges>

#include <CL/sycl.hpp>
#include <iostream>
#include "/opt/intel/oneapi/dev-utilities/latest/include/dpc_common.hpp"

using namespace sycl;

buffer<uint32_t> create_random_index_buffer( queue&, uint32_t );
buffer<uint32_t> create_fibonachi_buffer( queue&, uint32_t buffer_count );
bool check_randomization_buffer( queue& q, buffer<uint32_t>& input );
uint32_t get_sparse_index( uint32_t index, uint32_t all_count );

int main(int argc, char *argv[]) 
{
  try 
  {
		if( argc < 2 )
			return 1;

	auto buffer_size = std::stoul( argv[1] );
	auto thread_max = 100u;

	if( argc >= 3 )
		thread_max = std::stoul( argv[2] );

	uint32_t exact_test = 0;

	if( argc >= 4 )
		exact_test = std::stoul( argv[3] );

	sycl::queue q( sycl::default_selector(), dpc_common::exception_handler
		,property::queue::enable_profiling{} 
	);

	std::cout << "[SYCL] Using device: ["
		<< q.get_device().get_info<info::device::name>()
		<< "] from ["
		<< q.get_device().get_platform().get_info<info::platform::name>()
		<< "]\n";

	uint32_t all_reads( 134217728 );

	using buffer_type = uint32_t;
	auto buffer_count = buffer_size / sizeof(  buffer_type );

	constexpr uint32_t call_count{4000000000};

	uint32_t thread_counter = (exact_test == 1 || exact_test == 3 ) ? thread_max : 1u;
	uint32_t thread_shift = (exact_test == 1 || exact_test == 3 ) ? 30u : 0u;

	for( ;thread_counter <= thread_max; 
		( thread_counter == (1u << thread_shift) && thread_shift ) 
			? thread_counter += (1u << (thread_shift - 1u)) 
			: (thread_counter = (1u << ++thread_shift))
	)
	{
		uint32_t buffer_counter = (exact_test == 1 || exact_test == 2 ) ? buffer_count : 1u;
		uint32_t buffer_shift = (exact_test == 1 || exact_test == 2 ) ? 30u : 0u;

		for( ;	buffer_counter <= buffer_count;
		( buffer_counter == (1u << buffer_shift) && buffer_shift ) 
			? buffer_counter += (1u << (buffer_shift - 1u)) 
			: (buffer_counter = (1u << ++buffer_shift))
		)
		{
			buffer<uint32_t> write_buffer( thread_counter );


			auto infinite_permutation_buffer = create_random_index_buffer( q, buffer_counter );
			//check_randomization_buffer(  q, infinite_permutation_buffer );
				
			auto main_event = q.submit( [&]( handler& h )
			{
				auto in = infinite_permutation_buffer.get_access<access_mode::read>(h);
				auto out = write_buffer.get_access<access_mode::discard_write>(h);
				auto calls_pre_thread = call_count / thread_counter;

				h.parallel_for(
					write_buffer.get_range()
					,[=]( id<> id )
				{
					uint32_t index = 
						get_sparse_index( 
							(in.get_count() / thread_counter) * id[0]
							, in.get_count() 
						);
					uint32_t counter{0u};
							
					while( counter < calls_pre_thread )
					{
						index = in[index];	
						++counter;
					} 
							
					out[id] = index;
				});
			});

			//q.wait_and_throw();

			uint64_t time =
					main_event.get_profiling_info<info::event_profiling::command_end>()
					- main_event.get_profiling_info<info::event_profiling::command_start>();

			std::cout 
			<< "threads: " << thread_counter
			<< " , buffer_size: " << buffer_counter * sizeof( uint32_t ) << " ,time: " <<	time
			<< " ,latency: " << float(time) / call_count 
			<< " ,throughput: " << 
				buffer_counter * sizeof( uint32_t ) / float(time)
			<< " ,radix_throughput: " << 
				std::log2( buffer_counter * sizeof( uint32_t ) ) / float(time)

			<< std::endl;
		}
	}

	}
	catch (sycl::exception const& e) 
	{
		std::cout << "fail; synchronous exception occurred: " << e.what() << "\n";
		return -1;
	}
	return 0;
}

buffer<uint32_t> create_random_index_buffer( queue& q, uint32_t buffer_size )
{
  buffer<uint32_t> rv( buffer_size );

  namespace d = oneapi::dpl;
	namespace r = d::experimental::ranges;

	auto queue_policy = d::execution::make_device_policy(q);
	// r::copy( queue_policy, 
	// 	r::all_view( std::forward<BufferT>(input) )
	// 	,r::all_view< index_t, acm::discard_write >( rv )
	// );
  //d::fill( queue_policy, d::begin(have_index), d::end(have_index), false );

  q.submit( [&](handler& h) 
  {
    auto out = rv.get_access<access_mode::discard_write>(h);

    h.parallel_for(
      out.get_range()
      ,[=](id<> id)
    {
      
        out[get_sparse_index( id[0], out.get_count() )] = 
          get_sparse_index( (id[0] + 1u) % out.get_count(), out.get_count() );
    });
  });

  return rv;
}

uint32_t get_sparse_index( uint32_t index, uint32_t all_count )
{
  uint64_t shift_value{0u};
  uint8_t bit_counter{0u};

  while( index )
  {
    shift_value <<= 1u;
    shift_value += (index & 1u);
    ++bit_counter;
    index>>= 1u;
  }

  return ((all_count * shift_value - 1u) >> bit_counter) + 1u;
}

buffer<uint32_t> create_fibonachi_buffer( queue& q, uint32_t buffer_count )
{
	buffer< uint32_t > read_buffer( buffer_count );
	read_buffer.get_access<access_mode::discard_write>()[0] = 1 % buffer_count;

	if( buffer_count > 1 )
		read_buffer.get_access<access_mode::write>()[1] = 2 % buffer_count;

	q.submit( [&]( handler& h )
	{
		auto out = read_buffer.get_access<access_mode::write>(h);
		
		h.single_task(
			[=]()
			{
				for( uint32_t buffer_index = 2; buffer_index < out.get_count(); ++buffer_index )
				{
					out[ buffer_index ] = ( out[buffer_index - 1] + out[buffer_index - 2] ) % buffer_count;
				}
			}
		);
	});

  return read_buffer;
}

bool check_randomization_buffer( queue& q, buffer<uint32_t>& input )
{
	buffer<bool> exists( input.get_count() );
	buffer<uint32_t> first_index( 2u );

	q.fill( exists.get_access<access_mode::discard_write>().get_pointer(), false, input.get_count() );

	q.submit( [&]( handler& h ){
		auto e = exists.get_access<access_mode::read_write>(h);
		auto fi = first_index.get_access<access_mode::discard_write>(h);
		auto in = input.get_access<access_mode::read>(h);

		h.single_task(
			[=]()
			{
				uint32_t index{0u};
				uint32_t counter{0u};

				while( !e[index] )
				{
					e[index] = true;
					index = in[index];
					++counter;
				}

				fi[0] = counter;
				fi[1] = index;
			}
		);
	});

	auto fia = first_index.get_access<access_mode::read>();

	if( fia[0u] != input.get_count() )
		std::cout << fia[0u] << " " << fia[1u] << std::endl;

	return fia[0u] == input.get_count();
}