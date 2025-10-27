//Made by Han_feng

use crate::lib::Music_player::{Music_player, Sound_data};
use crate::lib::Vulkan_application::{Image_data, Vulkan_application};
use ffmpeg_next::format::sample::Type::Packed;
use ffmpeg_next::format::{input, Pixel, Sample};
use ffmpeg_next::frame::{Audio, Video};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::Flags;
use ffmpeg_next::software::{resampling, scaling};
use ffmpeg_next::Rational;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

//Consts
const TITLE: &str = "Rust Vulkan Video Player";

//Structs
pub struct Media_handler;

//Functional structs
#[derive(Default)]
pub struct States{
    pub start_decode: AtomicBool,
    pub ready: [AtomicBool; 2],
    pub start_video: AtomicBool,
    pub start_music: AtomicBool,
}

struct FPS_controler{
    avg_frame_interval: f64,
    last_time: Option<Instant>,
}

struct Wrapper<T>(T);

//Impls
impl Media_handler{
    pub fn new() -> Media_handler{
        ffmpeg_next::init().expect("Could not init ffmpeg!");

        Media_handler
    }

    pub fn run(&mut self, video_path: String){
        let states = States::new();
        let (frame_sender, frame_receiver) = mpsc::channel();
        let (sound_sender, sound_receiver) = mpsc::channel();

        let states = states;
        let frame_sender = frame_sender;
        let sound_sender = sound_sender;

        let mut context = input(&video_path).unwrap();

        let video_stream = context.streams().best(Type::Video).expect("Failed to get video stream");
        let audio_stream = context.streams().best(Type::Audio).expect("Failed to get audio stream");

        let video_stream_index = video_stream.index();
        let audio_stream_index = audio_stream.index();

        let video_context = ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters()).unwrap();
        let audio_context = ffmpeg_next::codec::context::Context::from_parameters(audio_stream.parameters()).unwrap();

        let mut video_decoder = video_context.decoder().video().unwrap();
        let mut audio_decoder = audio_context.decoder().audio().unwrap();

        let window_size = (video_decoder.width() as f64, video_decoder.height() as f64);

        //It's weird that scaling::Context doesn't impl Send but resampling::Context does...
        let mut frame_scaler = Wrapper(scaling::Context::get(
            video_decoder.format(),
            video_decoder.width(),
            video_decoder.height(),
            Pixel::RGBA,
            video_decoder.width(),
            video_decoder.height(),
            Flags::FAST_BILINEAR,
        ).unwrap());
        let mut sound_scaler = resampling::Context::get(
            audio_decoder.format(),
            audio_decoder.channel_layout(),
            audio_decoder.rate(),
            Sample::F32(Packed),
            audio_decoder.channel_layout(),
            audio_decoder.rate(),
        ).unwrap();

        let (frame_packet_sender, frame_packet_receiver) = mpsc::channel();
        let (sound_packet_sender, sound_packet_receiver) = mpsc::channel();

        let mut fps_controler = FPS_controler::new(video_stream.avg_frame_rate());
        let states_clone = states.clone();
        let _video_thread = thread::spawn(move || {
            let mut frame_scaler = frame_scaler;
            let states = states_clone;

            let mut raw_frame = Video::empty();
            let mut frame = Video::empty();
            let mut func = || {
                if let Ok(packet) = frame_packet_receiver.recv(){
                    video_decoder.send_packet(&packet).unwrap();

                    while video_decoder.receive_frame(&mut raw_frame).is_ok(){
                        frame_scaler.0.run(&raw_frame, &mut frame).unwrap();
                        fps_controler.update();
                        frame_sender.send(Image_data::new(&frame)).unwrap();
                    }
                }
            };

            //First synchronization
            func();
            states.ready[0].store(true, Ordering::Relaxed);
            while !states.ready[1].load(Ordering::Relaxed) {}
            states.start_video.store(true, Ordering::Relaxed);

            loop{
                func();
            }
        });

        let states_clone = states.clone();
        let _music_thread = thread::spawn(move || {
            let states = states_clone;

            let mut raw_sound = Audio::empty();
            let mut sound = Audio::empty();

            let mut func = || {
                if let Ok(packet) = sound_packet_receiver.recv(){
                    audio_decoder.send_packet(&packet).unwrap();

                    while audio_decoder.receive_frame(&mut raw_sound).is_ok(){
                        sound_scaler.run(&raw_sound, &mut sound).unwrap();
                        sound_sender.send(Sound_data::new(&sound)).unwrap();
                    }
                }
            };

            //First synchronization
            func();
            states.ready[1].store(true, Ordering::Relaxed);
            while !states.ready[0].load(Ordering::Relaxed) {}
            states.start_music.store(true, Ordering::Relaxed);

            loop{
                func();
            }
        });

        let states_clone = states.clone();
        let _decode_thread = thread::spawn(move || {
            let states = states_clone;
            while !states.start_decode.load(Ordering::Relaxed) {}

            for (stream, packet) in context.packets() {
                if stream.index() == video_stream_index{
                    frame_packet_sender.send(packet).unwrap();
                }

                else if stream.index() == audio_stream_index{
                    sound_packet_sender.send(packet).unwrap();
                }
            }
        });

        let mut video_player = Vulkan_application::new(TITLE.to_string(), window_size, frame_receiver, states.clone());
        let mut music_player = Music_player::new(sound_receiver, states.clone());

        music_player.run();
        video_player.run();
    }
}

impl States {
    pub fn new() -> Arc<States>{
        Arc::new(States::default())
    }
}

impl FPS_controler{
    fn new(avg_frame_rate: Rational) -> FPS_controler{
        FPS_controler{avg_frame_interval: avg_frame_rate.denominator() as f64 / avg_frame_rate.numerator() as f64, last_time: None}
    }

    fn update(&mut self){
        if let Some(last_time) = self.last_time{
            while last_time.elapsed().as_secs_f64() < self.avg_frame_interval{}
        }

        self.last_time = Some(Instant::now());
    }
}

unsafe impl<T> Send for Wrapper<T> {}
unsafe impl<T> Sync for Wrapper<T> {}