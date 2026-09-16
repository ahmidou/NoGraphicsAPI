#import <Cocoa/Cocoa.h>
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#import <QuartzCore/CAMetalLayer.h>
#include <stdio.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(strong) NSWindow* window;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)notification
{
    NSRect frame = NSMakeRect(100, 100, 1280, 720);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"NoGraphicsAPI Metal 4 — Clear Screen"];

    CAMetalLayer* metalLayer = [CAMetalLayer layer];
    [self.window.contentView setWantsLayer:YES];
    [self.window.contentView setLayer:metalLayer];
    metalLayer.contentsScale = self.window.backingScaleFactor;

    gpu::DeviceDesc desc{};
    desc.window = (__bridge void*)metalLayer;
    desc.swapchain_format = gpu::Format::bgra8_srgb;
    desc.desired_swapchain_image_count = 2;

    auto [device, error] = gpu::create_device(desc);
    if (error != gpu::Error::none)
    {
        fprintf(stderr, "Failed to create device\n");
        [NSApp terminate:nil];
        return;
    }

    auto* semaphore = gpu::create_timeline_semaphore(device);
    uint64_t frame_index = 0;

    printf("Device: %s\n", gpu::get_device_caps(device).device_name);
    [self.window makeKeyAndOrderFront:nil];

    // Run 60 frames then quit
    for (int i = 0; i < 60; ++i)
    {
        @autoreleasepool
        {
            [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.016]];

            auto frame = gpu::acquire(device);
            if (!frame.render_view)
                continue;

            auto* cmd = gpu::begin_commands(device);

            gpu::ColorAttachment color{};
            color.render_view = frame.render_view;
            color.load = gpu::LoadOp::clear;
            color.store = gpu::StoreOp::store;
            color.clear = { 0.2f, 0.3f, 0.8f, 1.0f };

            gpu::RenderingDesc render_desc{};
            render_desc.colors = { &color, 1 };

            gpu::begin_render_pass(cmd, render_desc);
            gpu::end_render_pass(cmd);

            ++frame_index;
            gpu::TimelinePoint completion{ semaphore, frame_index };
            gpu::submit_and_present(device, { &cmd, 1 }, completion);
        }
    }

    gpu::wait_idle(device);
    gpu::destroy_timeline_semaphore(semaphore);
    gpu::destroy_device(device);
    printf("Clean shutdown after 60 frames.\n");
    [NSApp terminate:nil];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender
{
    return YES;
}

@end

int main(int argc, const char* argv[])
{
    @autoreleasepool
    {
        NSApplication* app = [NSApplication sharedApplication];
        AppDelegate* delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}
