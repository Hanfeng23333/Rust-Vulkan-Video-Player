//Made by Han_feng

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Instant;
use ffmpeg_next::format::{input, Pixel};
use ffmpeg_next::frame::Video;
use ffmpeg_next::media::Type;
use ffmpeg_next::Rational;
use ffmpeg_next::software::scaling::{Context, Flags};
use vulkano::sync::now;
use crate::lib::Vulkan_application::{Image_data, Vulkan_application};

//Consts
const TITLE: &str = "Rust Vulkan Video Player";

pub struct Media_handler{
    //Video
    video_player: Vulkan_application,
    frame_sender: Sender<Image_data>
}

impl Media_handler{
    pub fn new() -> Media_handler{
        ffmpeg_next::init().unwrap();

        let (frame_sender, frame_receiver) = mpsc::channel();

        Media_handler{
            frame_sender,
            video_player: Vulkan_application::new(TITLE.to_string(), (1920.0, 1080.0), frame_receiver)
        }
    }

    pub fn run(&mut self, video_path: String){
        let states = self.video_player.states.clone();
        let frame_sender = self.frame_sender.clone();

        let decode_thread = thread::spawn(move || {
            let states = states;
            let frame_sender = frame_sender;
            let mut context = input(&video_path).unwrap();

            let video_stream = context.streams().best(Type::Video).expect("Failed to get video stream");
            let audio_stream = context.streams().best(Type::Audio).expect("Failed to get audio stream");

            let video_stream_index = video_stream.index();
            let audio_stream_index = audio_stream.index();

            let video_context = ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters()).unwrap();
            let audio_context = ffmpeg_next::codec::context::Context::from_parameters(audio_stream.parameters()).unwrap();

            let mut video_decoder = video_context.decoder().video().unwrap();
            let audio_decoder = audio_context.decoder().audio().unwrap();
            
            let mut fps_controler = FPS_controler::new(video_stream.avg_frame_rate());
            let mut frame_scaler = Context::get(
                video_decoder.format(),
                video_decoder.width(),
                video_decoder.height(),
                Pixel::RGBA,
                video_decoder.width(),
                video_decoder.height(),
                Flags::FAST_BILINEAR,
            ).unwrap();

            while !states.start_video.load(Ordering::Relaxed) {}

            for (stream, packet) in context.packets() {
                if stream.index() == video_stream_index{
                    let mut raw_frame = Video::empty();
                    let mut frame = Video::empty();
                    video_decoder.send_packet(&packet).unwrap();

                    while video_decoder.receive_frame(&mut raw_frame).is_ok(){
                        frame_scaler.run(&raw_frame, &mut frame).unwrap();
                        fps_controler.update();
                        frame_sender.send(Image_data{
                            width: frame.width(), height: frame.height(), data: frame.data(0).to_vec()
                        }).unwrap();
                        states.start_render.store(true, Ordering::Relaxed);
                    }
                }
            }
        });

        self.video_player.run();
    }
}

struct FPS_controler{
    avg_frame_interval: f64,
    last_time: Option<Instant>,
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