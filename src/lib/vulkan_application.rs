use std::collections::HashSet;
use std::sync::Arc;
use vulkano::command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::{Version, VulkanLibrary};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{MemoryAllocator, StandardMemoryAllocator};
use vulkano::swapchain::Surface;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

//Structs
pub struct Vulkan_application{
    //Window attributes
    window_attributes: WindowAttributes,
    window: Option<Arc<Window>>,
    
    //Vulkan attributes
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    surface: Option<Arc<Surface>>,
}

#[derive(Default)]
struct Queue_family_indices{
    graphics_family: Option<u32>,
}

//Impls
impl Vulkan_application{
    pub fn new(title:String) -> Vulkan_application{
        //Window attributes
        let window_attributes = WindowAttributes::default()
            .with_title(title);
        
        //Vulkan attributes that don't depend on the window
        let instance = get_vulkan_instance();
        let (physical_device, queue_family) = get_physical_device_and_indices(instance.clone());
        let (device, queues) = get_device_and_queues::<Vec<Arc<Queue>>>(physical_device.clone(), &queue_family);
        let graphics_queue = queues.iter().find(|q| q.queue_family_index() == queue_family.graphics_family.unwrap()).unwrap().clone();
        
        Vulkan_application{
            //Window
            window_attributes,
            window: None,
            
            //Vulkan
            instance, physical_device, device, graphics_queue,
            surface: None,
        }
    }
    
    pub fn run(&mut self){
        let event_loop = EventLoop::new().unwrap();
        event_loop.run_app(self).expect("Failed to run vulkan application");
    }
}

impl ApplicationHandler for Vulkan_application{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none(){
            
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        todo!()
    }
}

impl Queue_family_indices {
    fn new(physical_device: Arc<PhysicalDevice>) -> Queue_family_indices{
        let mut indices = Queue_family_indices::default();
        
        for (i, queue_family) in physical_device.queue_family_properties().iter().enumerate(){
            if queue_family.queue_flags.contains(QueueFlags::GRAPHICS){
                indices.graphics_family = Some(i as u32);
            }
            
            if indices.is_complete(){
                break;
            }
        }
        
        indices
    }
    
    fn is_complete(&self) -> bool{
        self.graphics_family.is_some()
    }
    
    fn collect_info<T>(&self) -> T
    where T: FromIterator<QueueCreateInfo>
    {
        [self.graphics_family]
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

//Tool functions
fn get_vulkan_instance() -> Arc<Instance>{
    Instance::new(
        VulkanLibrary::new().expect("Couldn't create Vulkan library"),
        InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            application_name: Some("Rust Vulkan Video Player".to_string()),
            application_version: Version{major:1, minor:0, patch:0},
            engine_name: Some("Hanfeng's Engine".to_string()),
            engine_version: Version{major:1, minor:0, patch:0},
            ..Default::default()
        }
    ).expect("Couldn't create Vulkan instance")
}

fn get_physical_device_and_indices(instance: Arc<Instance>) -> (Arc<PhysicalDevice>, Queue_family_indices){
    instance.enumerate_physical_devices().expect("Couldn't find physical devices")
        .filter_map(|physical_device| { 
            let indices = Queue_family_indices::new(physical_device.clone());
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
        }).expect("Couldn't find a suitable physical device")
}

fn get_device_and_queues<T>(physical_device: Arc<PhysicalDevice>, queue_family: &Queue_family_indices) -> (Arc<Device>, T)
where T: FromIterator<Arc<Queue>>
{
    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo{
            queue_create_infos: queue_family.collect_info(),
            ..Default::default()
        }
    ).expect("Couldn't create a device");

    (device, queues.collect())
}