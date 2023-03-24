<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis maxScale="0" version="3.22.11-Białowieża" styleCategories="AllStyleCategories" minScale="1e+08" hasScaleBasedVisibilityFlag="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal enabled="0" mode="0" fetchMode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option type="bool" name="WMSBackgroundLayer" value="false"/>
      <Option type="bool" name="WMSPublishDataSourceUrl" value="false"/>
      <Option type="int" name="embeddedWidgets/count" value="0"/>
      <Option type="QString" name="identify/format" value="Value"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option type="QString" name="name" value=""/>
      <Option name="properties"/>
      <Option type="QString" name="type" value="collection"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling enabled="false" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer nodataColor="" type="paletted" opacity="1" alphaBand="-1" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry label="1" value="1" color="#45d0a2" alpha="255"/>
        <paletteEntry label="2" value="2" color="#38f62e" alpha="255"/>
        <paletteEntry label="3" value="3" color="#60439e" alpha="255"/>
        <paletteEntry label="4" value="4" color="#4a8ae4" alpha="255"/>
        <paletteEntry label="5" value="5" color="#6485a7" alpha="255"/>
        <paletteEntry label="6" value="6" color="#078c62" alpha="255"/>
        <paletteEntry label="7" value="7" color="#7c0016" alpha="255"/>
        <paletteEntry label="8" value="8" color="#c18760" alpha="255"/>
        <paletteEntry label="9" value="9" color="#bd7304" alpha="255"/>
        <paletteEntry label="10" value="10" color="#a4a59e" alpha="255"/>
        <paletteEntry label="11" value="11" color="#dcdcdc" alpha="255"/>
        <paletteEntry label="12" value="12" color="#fbfff3" alpha="255"/>
        <paletteEntry label="13" value="13" color="#8e336b" alpha="255"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast gamma="1" contrast="0" brightness="0"/>
    <huesaturation saturation="0" invertColors="0" colorizeOn="0" colorizeBlue="128" grayscaleMode="0" colorizeGreen="128" colorizeStrength="100" colorizeRed="255"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
