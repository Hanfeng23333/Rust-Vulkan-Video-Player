//Made by Han_feng
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
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
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass, SubpassDescription};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::{acquire_next_image, ColorSpace, PresentFuture, PresentMode, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{GpuFuture, Sharing};
use vulkano::{shader, sync, Validated, Version, VulkanError, VulkanLibrary};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

//Structs
#[derive(Default)]
pub struct Vulkan_application{
    //Window attributes
    event_loop: Option<EventLoop<()>>,
    window_attributes: WindowAttributes,
    window: Option<Arc<Window>>,
    
    //Vulkan attributes
    instance: Option<Arc<Instance>>,
    _debug_messenger: Option<DebugUtilsMessenger>,
    physical_device: Option<Arc<PhysicalDevice>>,
    device: Option<Arc<Device>>,
    queues: HashMap<String, Option<Arc<Queue>>>,
    surface: Option<Arc<Surface>>,
    swap_chain: Option<Arc<Swapchain>>,
    swap_chain_images: Vec<Arc<Image>>,
    swap_chain_image_views: Vec<Arc<ImageView>>,
    swap_chain_frame_buffers: Vec<Arc<Framebuffer>>,
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    render_pass: Option<Arc<RenderPass>>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    fences: Vec<Option<Arc<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>>>>>, //Type placeholder `_` not allowed in item's signature [E0121] :(

    //Allocators
    memory_allocator: Option<Arc<dyn MemoryAllocator>>,
    command_buffer_allocator: Option<Arc<dyn CommandBufferAllocator>>,

    //Runtime attributes
    current_frame: usize,
}

#[derive(Default, Clone, Copy)]
struct Queue_family_indices{
    present_family: Option<u32>,
    graphics_family: Option<u32>,
}

//Impls
impl Vulkan_application{
    pub fn new(title: String, size:(f64, f64)) -> Vulkan_application{
        //Window attributes
        let event_loop = Some(EventLoop::new().expect("Failed to create event loop"));
        let window_attributes = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(size.0, size.1));

        //Vulkan attributes
        let instance = Some(get_vulkan_instance(event_loop.as_ref().unwrap()));
        
        Vulkan_application{
            //Window
            event_loop,
            window_attributes,

            //Vulkan
            instance,

            //Default
            ..Default::default()
        }
    }

    fn init_vulkan(&mut self) {
        let instance = self.instance.as_ref().unwrap().clone();
        let window = self.window.as_ref().unwrap().clone();

        //Surface
        let surface = Surface::from_window(instance.clone(), self.window.as_ref().unwrap().clone()).unwrap();
        self.surface = Some(surface.clone());

        //Debug messenger
        #[cfg(debug_assertions)]
        {
            self._debug_messenger = Some(DebugUtilsMessenger::new(instance.clone(), DebugUtilsMessengerCreateInfo{
                message_severity: DebugUtilsMessageSeverity::ERROR | DebugUtilsMessageSeverity::WARNING | DebugUtilsMessageSeverity::VERBOSE,
                message_type: DebugUtilsMessageType::GENERAL | DebugUtilsMessageType::PERFORMANCE | DebugUtilsMessageType::VALIDATION,
                ..DebugUtilsMessengerCreateInfo::user_callback(unsafe { DebugUtilsMessengerCallback::new(debug_callback) })
            }).unwrap());
        }

        //Physical Device
        let device_extensions = DeviceExtensions{
            khr_swapchain: true,
            ..Default::default()
        };
        let (physical_device, queue_family) = get_physical_device_and_indices(instance.clone(), surface.clone(), &device_extensions);
        self.physical_device = Some(physical_device.clone());

        //Device
        let (device, queues) = get_device_and_queues::<Vec<Arc<Queue>>>(physical_device.clone(), &queue_family, device_extensions);
        self.device = Some(device.clone());
        self.queues = queue_family.collect_dict().into_iter().map(|(name, i)|{
            (name, i.map(|family| queues.iter().find(|q| q.queue_family_index() == family).unwrap().clone()))
        }).collect();
        
        //Swap chain
        let (swap_chain, images) = get_swap_chain_and_images(physical_device.clone(), queue_family.clone(), device.clone(), surface.clone(), window.clone());
        self.swap_chain = Some(swap_chain.clone());
        self.swap_chain_images = images;
        self.swap_chain_image_views = self.swap_chain_images.iter().map(|image| {
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

        //Graphics pipeline
        let (graphics_pipeline, render_pass) = get_graphics_pipeline_and_render_pass(device.clone(), swap_chain.clone());
        self.graphics_pipeline = Some(graphics_pipeline.clone());
        self.render_pass = Some(render_pass.clone());

        //Frame buffers
        self.swap_chain_frame_buffers = self.swap_chain_image_views.iter().
            map(|image_view| Framebuffer::new(render_pass.clone(), FramebufferCreateInfo{
                attachments: vec![image_view.clone()],
                extent: swap_chain.image_extent(),
                layers: 1,
                ..Default::default()
            }).expect("Failed to create frame buffer")).collect();

        //Command buffer
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo::default()));
        self.command_buffer_allocator = Some(command_buffer_allocator.clone());
        self.command_buffers = get_command_buffers(command_buffer_allocator.clone(), device.clone(), &queue_family, &self.swap_chain_frame_buffers, graphics_pipeline.clone());

        //Sync objects
        self.fences = vec![None; MAX_FRAMES_IN_FLIGHT];
    }

    fn draw_frame(&mut self) {
        if let Some(fence) = self.fences[self.current_frame].clone(){
            fence.wait(None).unwrap();
        }

        let (index, suboptimal, acquire_future) = match acquire_next_image(self.swap_chain.as_ref().unwrap().clone(), None).map_err(Validated::unwrap){
            Ok(data) => data,
            Err(VulkanError::OutOfDate) => {
                return;
            }
            Err(e) => panic!("Failed to acquire next image: {}", e),
        };

        let future = match self.fences[self.current_frame].clone() {
            Some(fence) => fence.boxed(),
            None => {
                let mut current = sync::now(self.device.as_ref().unwrap().clone());
                current.cleanup_finished();
                current.boxed()
            }
        }
            .join(acquire_future)
            .then_execute(self.queues["graphics"].as_ref().unwrap().clone(), self.command_buffers[index as usize].clone()).unwrap()
            .then_swapchain_present(self.queues["graphics"].as_ref().unwrap().clone(), SwapchainPresentInfo::swapchain_image_index(self.swap_chain.as_ref().unwrap().clone(), index))
            .then_signal_fence_and_flush();

        self.fences[self.current_frame] = match future.map_err(Validated::unwrap){
            Ok(fence) => Some(Arc::new(fence)),
            Err(VulkanError::OutOfDate) => {
                None
            }
            Err(error) =>{
                eprintln!("Failed to acquire next image: {}", error);
                None
            }
        };

        self.current_frame = (self.current_frame+1)%MAX_FRAMES_IN_FLIGHT;
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

            self.init_vulkan();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
               self.draw_frame();
                self.window.as_ref().unwrap().request_redraw();
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

    fn collect_dict(&self) -> HashMap<String, Option<u32>>{
        [
            ("graphics".to_string(), self.graphics_family),
            ("present".to_string(), self.present_family)
        ].into_iter().collect()
    }
    
    fn collect_info<T>(&self) -> T
    where T: FromIterator<QueueCreateInfo>
    {
        HashSet::from([self.graphics_family, self.present_family])
            .into_iter().filter_map(|i| {
            if let Some(index) = i{
                Some(QueueCreateInfo{
                    queue_family_index: index,
                    ..Default::default()
                })
            }
            else{
                None
            }
        }).collect()
    }
}

impl IntoIterator for Queue_family_indices {
    type Item = u32;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut result = vec![];
        result.extend(self.graphics_family);
        result.extend(self.present_family);
        result.into_iter()
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

fn get_device_and_queues<T>(physical_device: Arc<PhysicalDevice>, queue_family: &Queue_family_indices, enabled_extensions: DeviceExtensions) -> (Arc<Device>, T)
where T: FromIterator<Arc<Queue>>
{
    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo{
            queue_create_infos: queue_family.collect_info(),
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
            min_image_count: (surface_capabilities.min_image_count+5).min(surface_capabilities.max_image_count.unwrap_or(surface_capabilities.min_image_count+5)),
            image_sharing: if indices.graphics_family != indices.present_family {
                Sharing::Concurrent(indices.into_iter().collect())
            }
            else{
                Sharing::Exclusive
            },
            pre_transform: surface_capabilities.current_transform,
            ..Default::default()
        }
    ).expect("Couldn't create a swap chain")
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
        ..Default::default()
    }).expect("Failed to create render pass");

    (GraphicsPipeline::new(
        device.clone(), None, GraphicsPipelineCreateInfo{
            stages: [
                "shaders/vert.spv",
                "shaders/frag.spv"
            ].into_iter().map(|shader_file| PipelineShaderStageCreateInfo::new(get_shader(shader_file.to_string(), device.clone()).entry_point("main").unwrap())).collect(),
            dynamic_state: [DynamicState::Viewport, DynamicState::Scissor].into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
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
            ..GraphicsPipelineCreateInfo::layout(PipelineLayout::new(device.clone(), PipelineLayoutCreateInfo {
                set_layouts: vec![],
                ..Default::default()
            }).expect("Failed to create shader layout"))
        }
    ).expect("Failed to create graphics pipeline"), render_pass)
}

fn get_command_buffers(allocator: Arc<dyn CommandBufferAllocator>, device: Arc<Device>, indices: &Queue_family_indices, frame_buffers: &[Arc<Framebuffer>], graphics_pipeline: Arc<GraphicsPipeline>) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    frame_buffers.iter().map(|frame_buffer| {
        let mut builder = AutoCommandBufferBuilder::primary(
            allocator.clone(), indices.graphics_family.unwrap(), CommandBufferUsage::MultipleSubmit).unwrap();

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
                .draw(3,1,0,0).unwrap()
                .end_render_pass(SubpassEndInfo::default()).unwrap();
        }

        builder.build().unwrap()
    }).collect()
}