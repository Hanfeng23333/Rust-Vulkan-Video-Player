//Made by Han_feng
use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::sampler::ComponentMapping;
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage, SampleCount};
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexBuffersCollection, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass, SubpassDependency, SubpassDescription};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::{acquire_next_image, ColorSpace, PresentFuture, PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{AccessFlags, GpuFuture, PipelineStages, Sharing};
use vulkano::{shader, sync, Validated, Version, VulkanError, VulkanLibrary};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

//Consts
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const IMAGE_EXTENSION: u32 = 2;
#[cfg(debug_assertions)]
const FPS_COUNT_INTERVAL: u64 = 5;

//Vertex resources
const VERTEX_DATA: [Vertex_data; 4] = [
    Vertex_data{position: [-0.5, -0.5], color: [1.0, 0.0, 0.0]},
    Vertex_data{position: [0.5, -0.5], color: [0.0, 1.0, 0.0]},
    Vertex_data{position: [0.5, 0.5], color: [0.0, 0.0, 1.0]},
    Vertex_data{position: [-0.5, 0.5], color: [1.0, 1.0, 1.0]},
];
const INDICES:[u16; 6] = [0, 1, 2, 2, 3, 0];

//Structs
pub struct Vulkan_application{
    //Window attributes
    event_loop: Option<EventLoop<()>>,
    window_attributes: WindowAttributes,
    window: Option<Arc<Window>>,
    
    //Vulkan attributes
    instance: Arc<Instance>,
    _debug_messenger: Option<DebugUtilsMessenger>,
    render_context: Option<Render_context>,
}

#[derive(Default, Clone)]
struct Queue_family_indices{
    graphics_family: Option<u32>,
    present_family: Option<u32>,

    //Iterator
    elements: Option<VecDeque<u32>>,
}

struct Render_context{
    physical_device: Arc<PhysicalDevice>,
    indices: Queue_family_indices,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
    surface: Arc<Surface>,
    swap_chain: Arc<Swapchain>,
    swap_chain_images: Vec<Arc<Image>>,
    swap_chain_image_views: Vec<Arc<ImageView>>,
    frame_buffers: Vec<Arc<Framebuffer>>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    vertex_buffer: Subbuffer<[Vertex_data]>,
    index_buffer: Subbuffer<[u16]>,

    //Allocators
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,

    //Runtime attributes
    current_frame: usize,
    refresh_swap_chain: bool,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    fences: Vec<Option<Arc<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>>>>, //Type placeholder `_` not allowed in item's signature [E0121] :(
    #[cfg(debug_assertions)] fps_counter: FPS_counter,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct Vertex_data{
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

#[cfg(debug_assertions)]
struct FPS_counter{
    fps: u64,
    previous_time: Instant,
}

//Impls
impl Vulkan_application{
    pub fn new(title: String, size:(f64, f64)) -> Vulkan_application{
        //Window attributes
        let event_loop = Some(EventLoop::new().expect("Failed to create event loop"));

        //Vulkan attributes
        let instance = get_vulkan_instance(event_loop.as_ref().unwrap());
        let _debug_messenger = if cfg!(debug_assertions){
            Some(DebugUtilsMessenger::new(instance.clone(), DebugUtilsMessengerCreateInfo{
                message_severity: DebugUtilsMessageSeverity::ERROR | DebugUtilsMessageSeverity::WARNING | DebugUtilsMessageSeverity::VERBOSE,
                message_type: DebugUtilsMessageType::GENERAL | DebugUtilsMessageType::PERFORMANCE | DebugUtilsMessageType::VALIDATION,
                ..DebugUtilsMessengerCreateInfo::user_callback(unsafe { DebugUtilsMessengerCallback::new(debug_callback) })
            }).unwrap())
        }
        else{
            None
        };
        
        Vulkan_application{
            //Window
            event_loop,
            window: None,
            window_attributes: WindowAttributes::default()
                .with_title(title)
                .with_inner_size(LogicalSize::new(size.0, size.1)),

            //Vulkan
            instance, _debug_messenger,
            render_context: None,
        }
    }

    pub fn run(&mut self){
        let event_loop = self.event_loop.take().expect("Failed to get event loop");
        event_loop.run_app(self).expect("Failed to run vulkan application");
    }
}

impl ApplicationHandler for Vulkan_application{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none(){
            let window = Arc::new(event_loop.create_window(self.window_attributes.clone()).expect("Failed to create window"));
            self.window = Some(window.clone());
            self.render_context = Some(Render_context::new(self.instance.clone(), window.clone()));
            self.render_context.as_mut().unwrap().fps_counter.reset();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            WindowEvent::Resized(_) => {
                self.render_context.as_mut().unwrap().refresh_swap_chain = true;
            }
            WindowEvent::RedrawRequested => {
                let render_context = self.render_context.as_mut().unwrap();
                let window = self.window.as_ref().unwrap();

                render_context.draw_frame(window.clone());

                #[cfg(debug_assertions)]
                {
                    if let Some(fps) = render_context.fps_counter.update(FPS_COUNT_INTERVAL){
                        println!("FPS: {}", fps);
                    }
                }

                window.request_redraw();
            }
            _ => ()
        }
    }
}

impl Queue_family_indices {
    fn new(physical_device: Arc<PhysicalDevice>, surface: Arc<Surface>) -> Queue_family_indices{
        let mut indices = Queue_family_indices::default();
        
        for (i, queue_family) in physical_device.queue_family_properties().iter().enumerate(){
            if queue_family.queue_flags.contains(QueueFlags::GRAPHICS){
                indices.graphics_family = Some(i as u32);
            }

            if physical_device.surface_support(i as u32, &surface).unwrap_or(false){
                indices.present_family = Some(i as u32);
            }
            
            if indices.is_complete(){
                break;
            }
        }
        
        indices
    }
    
    fn is_complete(&self) -> bool{
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

impl Iterator for Queue_family_indices {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.elements.is_none(){
            self.elements = Some([self.graphics_family, self.present_family].into_iter().flatten().collect());
        }

        self.elements.as_mut().unwrap().pop_front()
    }
}

impl Render_context {
    fn new(instance: Arc<Instance>, window: Arc<Window>) -> Render_context{
        //Surface
        let surface = Surface::from_window(instance.clone(), window.clone()).expect("Failed to create surface");

        //Physical Device
        let device_extensions = DeviceExtensions{
            khr_swapchain: true,
            ..Default::default()
        };
        let (physical_device, indices) = get_physical_device_and_indices(instance.clone(), surface.clone(), &device_extensions);

        //Device
        let (device, queues) = get_device_and_queues::<Vec<Arc<Queue>>>(physical_device.clone(), indices.clone(), device_extensions);

        //Queues
        let graphics_queue = queues.iter().find(|q| q.queue_family_index() == indices.graphics_family.unwrap()).unwrap().clone();
        let present_queue = queues.iter().find(|q| q.queue_family_index() == indices.present_family.unwrap()).unwrap().clone();

        //Swap chain
        let (swap_chain, swap_chain_images) = get_swap_chain_and_images(physical_device.clone(), indices.clone(), device.clone(), surface.clone(), window.clone());

        //Graphics pipeline
        let (graphics_pipeline, render_pass) = get_graphics_pipeline_and_render_pass(device.clone(), swap_chain.clone());

        //Image views and frame buffers
        let (swap_chain_image_views, frame_buffers) = get_image_views_and_frame_buffers(swap_chain.clone(), &swap_chain_images, render_pass.clone());

        //Allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default()));

        //Buffers
        let vertex_buffer = Buffer::from_iter(memory_allocator.clone(), BufferCreateInfo{
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        }, AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        }, VERTEX_DATA).expect("Failed to create vertex buffer");
        let index_buffer = Buffer::from_iter(memory_allocator.clone(), BufferCreateInfo{
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        }, AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        }, INDICES).expect("Failed to create index buffer");
        let command_buffers:Vec<_> = frame_buffers.iter().map(|frame_buffer| get_command_buffer(command_buffer_allocator.clone(), CommandBufferUsage::MultipleSubmit, &indices, frame_buffer.clone(), graphics_pipeline.clone(), vertex_buffer.clone(), index_buffer.clone())).collect();

        Render_context{
            physical_device, indices, device, graphics_queue, present_queue, surface, swap_chain, swap_chain_images, swap_chain_image_views, frame_buffers, graphics_pipeline, render_pass, memory_allocator, command_buffer_allocator, vertex_buffer, index_buffer, command_buffers,
            fences: vec![None; MAX_FRAMES_IN_FLIGHT],
            current_frame: 0,
            refresh_swap_chain: false,
            fps_counter: FPS_counter::new()
        }
    }

    fn recreate_swap_chain(&mut self, window: Arc<Window>) {
        unsafe { self.device.wait_idle().expect("Failed to wait on device idle"); }

        //Swap chain
        let (swap_chain, images) = self.swap_chain.recreate(SwapchainCreateInfo{
            image_extent: window.inner_size().into(),
            ..self.swap_chain.create_info()
        }).expect("Failed to recreate swap chain image");
        self.swap_chain = swap_chain.clone();
        self.swap_chain_images = images;

        let (image_views, frame_buffers) = get_image_views_and_frame_buffers(swap_chain.clone(), &self.swap_chain_images, self.render_pass.clone());
        self.swap_chain_image_views = image_views;
        self.command_buffers = frame_buffers.iter().map(|frame_buffer| get_command_buffer(self.command_buffer_allocator.clone(), CommandBufferUsage::MultipleSubmit, &self.indices, frame_buffer.clone(), self.graphics_pipeline.clone(), self.vertex_buffer.clone(), self.index_buffer.clone())).collect();
        self.frame_buffers = frame_buffers;
    }

    fn draw_frame(&mut self, window: Arc<Window>) {
        if self.refresh_swap_chain{
            self.refresh_swap_chain = false;
            self.recreate_swap_chain(window);
        }

        if let Some(fence) = self.fences[self.current_frame].clone(){
            fence.wait(None).unwrap();
        }

        let (index, suboptimal, acquire_future) = match acquire_next_image(self.swap_chain.clone(), None).map_err(Validated::unwrap){
            Ok(data) => data,
            Err(VulkanError::OutOfDate) => {
                return;
            }
            Err(e) => panic!("Failed to acquire next image: {}", e),
        };

        if suboptimal{
            self.refresh_swap_chain = true;
            return;
        }

        let future = match self.fences[self.current_frame].clone() {
            Some(fence) => fence.boxed(),
            None => {
                let mut current = sync::now(self.device.clone());
                current.cleanup_finished();
                current.boxed()
            }
        }
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), self.command_buffers[index as usize].clone()).unwrap()
            .then_swapchain_present(self.present_queue.clone(), SwapchainPresentInfo::swapchain_image_index(self.swap_chain.clone(), index))
            .then_signal_fence_and_flush();

        self.fences[self.current_frame] = match future.map_err(Validated::unwrap){
            Ok(fence) => Some(Arc::new(fence)),
            Err(VulkanError::OutOfDate) => {
                self.refresh_swap_chain = true;
                None
            }
            Err(error) =>{
                eprintln!("Failed to acquire next image: {}", error);
                None
            }
        };

        self.current_frame = (self.current_frame+1)%MAX_FRAMES_IN_FLIGHT;
    }
}

#[cfg(debug_assertions)]
impl FPS_counter {
    fn new() -> FPS_counter{
        FPS_counter{fps: 0, previous_time: Instant::now()}
    }

    fn reset(&mut self){
        self.previous_time = Instant::now();
    }

    fn update(&mut self, interval: u64) -> Option<u64> {
        self.fps += 1;
        if self.previous_time.elapsed().as_secs() >= interval {
            self.previous_time = Instant::now();
            let result = Some(self.fps/interval);
            self.fps = 0;
            result
        }
        else {
            None
        }
    }
}

//Tool functions
fn get_vulkan_instance(event_loop: &EventLoop<()>) -> Arc<Instance>{
    let library = VulkanLibrary::new().expect("Couldn't create Vulkan library");

    let mut enabled_layers = vec![];
    let mut enabled_extensions = Surface::required_extensions(event_loop).unwrap();
    enabled_layers.extend(get_validation_layers(library.clone(), &mut enabled_extensions));
    
    Instance::new(
        library,
        InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            application_name: Some("Rust Vulkan Video Player".to_string()),
            application_version: Version{major:1, minor:0, patch:0},
            engine_name: Some("Hanfeng's Engine".to_string()),
            engine_version: Version{major:1, minor:0, patch:0},
            enabled_layers,
            enabled_extensions,
            ..Default::default()
        }
    ).expect("Couldn't create Vulkan instance")
}

fn get_validation_layers(library: Arc<VulkanLibrary>, instance_extensions: &mut InstanceExtensions) -> Vec<String>{
    #[cfg(debug_assertions)]
    {
        let validation_layers = vec![
            "VK_LAYER_KHRONOS_validation".to_string()
        ];

        let available_layers:HashSet<String> = library.layer_properties().expect("Failed to get layer properties").map(|layer| layer.name().to_string()).collect();

        let result:Vec<String> = validation_layers.into_iter().filter(|layer| available_layers.contains(layer)).collect();
        if result.is_empty(){
            println!("No validation layers available");
        }
        else{
            instance_extensions.ext_debug_utils = true;
            println!("Validation layers:");
            for layer in result.iter(){
                println!("{}", layer);
            }
        }

        result
    }
    #[cfg(not(debug_assertions))]
    {
        vec![]
    }
}

#[cfg(debug_assertions)]
fn debug_callback(message_severity:DebugUtilsMessageSeverity, message_type:DebugUtilsMessageType, callback_data:DebugUtilsMessengerCallbackData){
    let message = format!("Validation Layer[{:?}][{:?}]: {}", message_severity, message_type, callback_data.message);
    if message_severity.intersects(DebugUtilsMessageSeverity::ERROR){
        eprintln!("{}", message);
    }
    else{
        println!("{}", message);
    }
}

fn get_physical_device_and_indices(instance: Arc<Instance>, surface: Arc<Surface>, device_extensions: &DeviceExtensions) -> (Arc<PhysicalDevice>, Queue_family_indices){
    let device_pack = instance.enumerate_physical_devices().expect("Couldn't find physical devices")
        .filter(|physical_device|
            physical_device.supported_extensions().contains(device_extensions)
            && !physical_device.surface_formats(&surface, Default::default()).unwrap().is_empty()
            && !physical_device.surface_present_modes(&surface, Default::default()).unwrap().is_empty()
        )
        .filter_map(|physical_device| {
            let indices = Queue_family_indices::new(physical_device.clone(), surface.clone());
            if indices.is_complete(){
                Some((physical_device, indices))
            }
            else { 
                None
            }
        })
        .min_by_key(|(physical_device, _)| match physical_device.properties().device_type { 
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4
        }).expect("Couldn't find a suitable physical device");

    #[cfg(debug_assertions)]
    {
        println!("Using physical device: {}", device_pack.0.properties().device_name);
    }

    device_pack
}

fn get_device_and_queues<T>(physical_device: Arc<PhysicalDevice>, indices: Queue_family_indices, enabled_extensions: DeviceExtensions) -> (Arc<Device>, T)
where T: FromIterator<Arc<Queue>>
{
    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo{
            queue_create_infos: indices.collect::<HashSet<_>>()
                .into_iter().map(|i| {
                QueueCreateInfo{
                    queue_family_index: i,
                    ..Default::default()
                }
            }).collect(),
            enabled_extensions,
            ..Default::default()
        }
    ).expect("Couldn't create a device");

    (device, queues.collect())
}

fn get_swap_chain_and_images(physical_device: Arc<PhysicalDevice>, indices: Queue_family_indices, device: Arc<Device>, surface: Arc<Surface>, window: Arc<Window>) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    //Choose a surface format
    let surface_formats = physical_device.surface_formats(&surface, Default::default()).unwrap();
    let (image_format, image_color_space) = if surface_formats.contains(&(Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear)){
        (Format::B8G8R8A8_SRGB, ColorSpace::SrgbNonLinear)
    }
    else{
        surface_formats[0]
    };

    //Choose a present mode
    let present_modes = physical_device.surface_present_modes(&surface, Default::default()).unwrap();
    let present_mode = if present_modes.contains(&PresentMode::Mailbox){
        PresentMode::Mailbox
    }
    else{
        PresentMode::Fifo
    };

    //Make a swap extent
    let surface_capabilities = physical_device.surface_capabilities(&surface, Default::default()).unwrap();
    let image_extent = if let Some(extent) = surface_capabilities.current_extent && extent[0] != u32::MAX{
        extent
    }
    else{
        let [width, height]:[u32;2] = window.inner_size().into();
        let [min_width, min_height] = surface_capabilities.min_image_extent;
        let [max_width, max_height] = surface_capabilities.max_image_extent;
        [width.clamp(min_width, max_width), height.clamp(min_height, max_height)]
    };

    Swapchain::new(
        device, surface, SwapchainCreateInfo{
            image_format, image_color_space, present_mode, image_extent,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            min_image_count: (surface_capabilities.min_image_count+IMAGE_EXTENSION).min(surface_capabilities.max_image_count.unwrap_or(surface_capabilities.min_image_count+IMAGE_EXTENSION)),
            image_sharing: if indices.graphics_family != indices.present_family {
                Sharing::Concurrent(indices.collect())
            }
            else{
                Sharing::Exclusive
            },
            pre_transform: surface_capabilities.current_transform,
            ..Default::default()
        }
    ).expect("Couldn't create a swap chain")
}

fn get_image_views_and_frame_buffers(swap_chain: Arc<Swapchain>, images: &[Arc<Image>], render_pass: Arc<RenderPass>) -> (Vec<Arc<ImageView>>, Vec<Arc<Framebuffer>>) {
    let image_views:Vec<_> = images.iter().map(|image| {
        ImageView::new(image.clone(), ImageViewCreateInfo{
            format: swap_chain.image_format(),
            component_mapping: ComponentMapping::identity(),
            subresource_range: ImageSubresourceRange{
                aspects: ImageAspects::COLOR,
                mip_levels: 0..1,
                array_layers: 0..1
            },
            ..Default::default()
        }).expect("Failed to create swap chain image view")
    }).collect();

    let frame_buffers = image_views.iter().
        map(|image_view| Framebuffer::new(render_pass.clone(), FramebufferCreateInfo{
            attachments: vec![image_view.clone()],
            extent: swap_chain.image_extent(),
            layers: 1,
            ..Default::default()
        }).expect("Failed to create frame buffer")).collect();

    (image_views, frame_buffers)
}

fn get_shader(file_path: String, device: Arc<Device>) -> Arc<ShaderModule>{
    let mut shader_file = File::open(file_path).unwrap();
    let mut shader_data = vec![];
    shader_file.read_to_end(&mut shader_data).unwrap();
    
    unsafe {
        ShaderModule::new(device, ShaderModuleCreateInfo::new(&shader::spirv::bytes_to_words(&shader_data).expect("Couldn't read spirv shader data")))
            .expect("Failed to create shader module")
    }
}
fn get_graphics_pipeline_and_render_pass(device: Arc<Device>, swap_chain: Arc<Swapchain>) -> (Arc<GraphicsPipeline>, Arc<RenderPass>) {
    //Render pass
    let render_pass = RenderPass::new(device.clone(), RenderPassCreateInfo{
        attachments: vec![AttachmentDescription{
            format: swap_chain.image_format(),
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::PresentSrc,
            ..Default::default()
        }],
        subpasses: vec![SubpassDescription{
            color_attachments: vec![Some(AttachmentReference{
                layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            })],
            ..Default::default()
        }],
        dependencies: vec![SubpassDependency{
            dst_subpass: Some(0),
            src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }],
        ..Default::default()
    }).expect("Failed to create render pass");

    //Shaders
    let vertex_shader = get_shader("shaders/vert.spv".to_string(), device.clone());
    let fragment_shader = get_shader("shaders/frag.spv".to_string(), device.clone());

    let vertex_entry = vertex_shader.entry_point("main").unwrap();
    let fragment_entry = fragment_shader.entry_point("main").unwrap();

    let vertex_input_state = Vertex_data::per_vertex().definition(&vertex_entry).unwrap();

    let stages = [
        vertex_entry, fragment_entry,
    ].into_iter().map(|shader_entry| PipelineShaderStageCreateInfo::new(shader_entry)).collect();

    //Layout
    let layout = PipelineLayout::new(device.clone(), PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        .into_pipeline_layout_create_info(device.clone()).unwrap()).unwrap();

    //Pipeline
    (GraphicsPipeline::new(
        device.clone(), None, GraphicsPipelineCreateInfo{
            stages,
            dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState{
                cull_mode: CullMode::Back,
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState{
                attachments: vec![ColorBlendAttachmentState{
                    blend: Some(AttachmentBlend{
                        dst_alpha_blend_factor: BlendFactor::Zero,
                        ..AttachmentBlend::alpha()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        }
    ).expect("Failed to create graphics pipeline"), render_pass)
}

fn get_command_buffer(allocator: Arc<dyn CommandBufferAllocator>, usage: CommandBufferUsage, indices: &Queue_family_indices, frame_buffer: Arc<Framebuffer>, graphics_pipeline: Arc<GraphicsPipeline>, vertex_buffer: impl VertexBuffersCollection, index_buffer: impl Into<IndexBuffer>) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        allocator.clone(), indices.graphics_family.unwrap(), usage).unwrap();

    //Command record
    unsafe {
        builder
            .begin_render_pass(RenderPassBeginInfo{
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(frame_buffer.clone())
            }, SubpassBeginInfo{
                contents: SubpassContents::Inline,
                ..Default::default()
            }).unwrap()
            .bind_pipeline_graphics(graphics_pipeline.clone()).unwrap()
            .set_viewport(0, [Viewport{
                extent: frame_buffer.extent().map(|i| i as f32),
                ..Default::default()
            }
            ].into_iter().collect()).unwrap()
            .set_scissor(0, [Scissor{
                extent: frame_buffer.extent(),
                ..Default::default()
            }].into_iter().collect()).unwrap()
            .bind_vertex_buffers(0, vertex_buffer).unwrap()
            .bind_index_buffer(index_buffer).unwrap()
            .draw_indexed(INDICES.len() as u32,1,0,0, 0).unwrap()
            .end_render_pass(SubpassEndInfo::default()).unwrap();
    }

    builder.build().unwrap()
}