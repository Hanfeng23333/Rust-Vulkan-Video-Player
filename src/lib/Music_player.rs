//Made by Han_feng

use std::sync::Arc;
use std::sync::atomic::Ordering;
use rodio::mixer::Mixer;
use rodio::{ChannelCount, OutputStream, SampleRate, Sink, Source};
use std::sync::mpsc::Receiver;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use ffmpeg_next::frame::Audio;
use crate::lib::Media_handler::States;

//Structs
pub struct Music_player{
    music_stream: OutputStream,
    music_context: Option<Music_context>
}

pub struct Sound_data{
    channels: u16,
    sample_rate: u32,
    duration: Duration,

    //Iterator
    data: Vec<f32>,
    index: usize,
}

struct Music_context{
    sink: Sink,
    sound_receiver: Receiver<Sound_data>,
    states: Arc<States>
}

//Impls
impl Music_player {
    pub fn new(sound_receiver: Receiver<Sound_data>, states: Arc<States>) -> Music_player {
        let music_stream = rodio::OutputStreamBuilder::open_default_stream().expect("Could not open music stream");

        Music_player{
            music_context: Some(Music_context::new(&music_stream.mixer(), sound_receiver, states)),
            music_stream
        }
    }

    pub fn run(&mut self) -> JoinHandle<()>{
        let music_context = self.music_context.take().unwrap();

        thread::spawn(move || {
            let music_context = music_context;

            while !music_context.states.start_music.load(Ordering::Relaxed) {}

            loop{
                if let Ok(sound_data) = music_context.sound_receiver.recv() {
                    music_context.sink.append(sound_data);
                }
            }
        })
    }
}

impl Sound_data {
    pub fn new(sound: &Audio) -> Sound_data{
        Sound_data{
            channels: sound.channels(),
            sample_rate: sound.rate(),
            duration: Duration::from_secs_f64(sound.samples() as f64 / sound.rate() as f64),
            data: unsafe{
                let data = sound.data(0).as_ptr() as *const f32;
                std::slice::from_raw_parts(data, sound.samples()*sound.channels() as usize).to_vec()
            },
            index: 0,
        }
    }
}

impl Iterator for Sound_data {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            self.index += 1;
            Some(self.data[self.index-1])
        }
        else {
            None
        }
    }
}

impl Source for Sound_data{
    fn current_span_len(&self) -> Option<usize> {
        Some(self.data.len()-self.index)
    }

    fn channels(&self) -> ChannelCount {
        self.channels
    }

    fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(self.duration)
    }
}

impl Music_context {
    fn new(mixer: &Mixer, sound_receiver: Receiver<Sound_data>, states: Arc<States>) -> Music_context {
        let sink = Sink::connect_new(mixer);

        Music_context{
            sink, sound_receiver, states
        }
    }
}