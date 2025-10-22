//Made by Han_feng
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use smallvec::smallvec;
use vulkano::command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::{Version, VulkanLibrary};
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackData, DebugUtilsMessengerCreateInfo};
use vulkano::memory::allocator::{MemoryAllocator, StandardMemoryAllocator};
use vulkano::swapchain::{ColorSpace, PresentMode, Surface, SurfaceCapabilities, Swapchain, SwapchainCreateFlags, SwapchainCreateInfo};
use vulkano::sync::Sharing;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

//Structs
#[derive(Default)]
pub struct Vulkan_application{
    //Window attributes
    event_loop: Option<EventLoop<()>>,
    window_attributes: WindowAttributes,
    window: Option<Arc<Window>>,
    
    //Vulkan attributes
    instance: Option<Arc<Instance>>,
    debug_messenger: Option<DebugUtilsMessenger>,
    physical_device: Option<Arc<PhysicalDevice>>,
    device: Option<Arc<Device>>,
    queues: HashMap<String, Option<Arc<Queue>>>,
    surface: Option<Arc<Surface>>,
    swap_chain: Option<Arc<Swapchain>>,
    swap_chain_images: Option<Vec<Arc<Image>>>
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
            self.debug_messenger = Some(DebugUtilsMessenger::new(instance.clone(), DebugUtilsMessengerCreateInfo{
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
        let (swap_chain, images) = get_swap_chain_and_images(physical_device.clone(), &queue_family, device.clone(), surface.clone(), window.clone());
        self.swap_chain = Some(swap_chain.clone());
        self.swap_chain_images = Some(images);
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
            ("graphics_family".to_string(), self.graphics_family),
            ("present_family".to_string(), self.present_family)
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
    type IntoIter = smallvec::IntoIter<Self::Item, 4>;

    fn into_iter(self) -> Self::IntoIter {
        let mut result = smallvec![];
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

fn get_swap_chain_and_images(physical_device: Arc<PhysicalDevice>, indices: &Queue_family_indices, device: Arc<Device>, surface: Arc<Surface>, window: Arc<Window>) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
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
                Sharing::Concurrent(indices.clone().into_iter().collect())
            }
            else{
                Sharing::Exclusive
            },
            pre_transform: surface_capabilities.current_transform,
            clipped: true,
            ..Default::default()
        }
    ).expect("Couldn't create a swap chain")
}